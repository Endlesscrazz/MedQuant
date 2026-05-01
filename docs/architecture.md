# MedQuant — Architecture
# Written: 2026-04-27
# Purpose: module specs, interface contracts, data flow, eval harness design,
#          results schema, and CB integration points.
# ──────────────────────────────────────────────────────────────────────────

## SYSTEM OVERVIEW

```
PubMedQA + MedMCQA (HuggingFace)
         │
         ▼
  src/data/loader.py          Load raw QA pairs, cache to disk
         │
         ▼
  src/data/formatter.py       Format as instruction-tuning pairs
         │                    (chat template applied per model)
         ▼
  src/train/train.py          QLoRA fine-tuning (reads config/*.yaml)
         │
         ▼
  outputs/checkpoints/        LoRA adapter weights (CHPC scratch only)
         │
         ▼
  src/convert/merge_lora.py   Merge adapter into base model (BF16)
         │
         ▼
  src/convert/to_gguf.py      llama.cpp conversion at 5 quantization levels
         │
         ▼
  outputs/gguf/               10 GGUF files (5 base + 5 fine-tuned)
         │
         ▼
  src/eval/batch_eval.py      Run all 10 variants on PubMedQA + MedMCQA
         │
         ▼
  metrics/results.json        Full results table (accuracy, size, latency)
         │
         ▼
  src/analysis/charts.py      Quality cliff chart + efficiency frontier
         │
         ▼
  src/serve/app.py            FastAPI endpoint (Q4_K_M fine-tuned, local Mac)
```

---

## DIRECTORY STRUCTURE

```
MedQuant/
├── CLAUDE.md
├── .gitignore                  # Excludes weights, GGUF, scratch paths
├── pyproject.toml
├── config/
│   ├── gpu_config.yaml         # Llama-3.1-8B, A800 settings
│   └── mac_config.yaml         # Qwen-2.5-1.5B, MPS settings
├── src/
│   ├── common/
│   │   └── utils.py            # Shared utilities (wilson CI, file-lock append)
│   ├── data/
│   │   ├── loader.py           # HuggingFace dataset loading
│   │   ├── formatter.py        # Instruction format + chat template
│   │   └── contamination_check.py  # PMID-based deduplication for PubMedQA
│   ├── train/
│   │   ├── train.py            # QLoRA training script (entry point)
│   │   └── lora_config.py      # LoRA/QLoRA config builder from yaml
│   ├── eval/
│   │   ├── eval.py             # Single model eval runner (logprob scoring)
│   │   ├── metrics.py          # Accuracy, CIs, drop_from_f16, brittleness
│   │   ├── batch_eval.py       # Runs one variant, writes metrics/raw/{variant}.json
│   │   └── merge_results.py    # Merges 10 raw files → metrics/results.json
│   ├── convert/
│   │   ├── merge_lora.py       # LoRA merge → BF16 weights
│   │   └── to_gguf.py          # Wraps llama.cpp convert + quantize
│   ├── analysis/
│   │   └── charts.py           # Quality cliff + efficiency frontier plots
│   └── serve/
│       └── app.py              # FastAPI endpoint, llama-cpp-python backend
├── slurm/
│   ├── train.sbatch
│   ├── eval.sbatch
│   ├── convert.sbatch
│   └── launch_eval.sh      # parallel per-variant eval launcher
├── tests/
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_serve.py
├── verify/
│   ├── HOW_TO_VERIFY.md
│   ├── check_cb.sh
│   ├── eval_cb_packet.py
│   └── ground_truth/
├── metrics/
│   ├── raw/                    # per-variant raw eval outputs (one file per SLURM job)
│   ├── results.json            # merged final table (derived from raw/)
│   ├── charts/
│   └── versions.lock           # reproducibility record — committed once
└── docs/
```

---

## MODULE SPECS

### src/data/loader.py

Responsibility: load PubMedQA and MedMCQA from HuggingFace or local cache.
Supports TRANSFORMERS_OFFLINE=1 for CHPC compute nodes.

Public interface:
```python
def load_dataset(
    name: Literal["pubmedqa", "medmcqa"],
    split: Literal["train", "validation", "test"],
    cache_dir: str,
    max_samples: int | None = None,   # mac fallback subsampling
) -> list[dict]
```

Returns standardized dicts:
  pubmedqa: {pubid, question, context, answer, task="pubmedqa"}
  medmcqa:  {question, opa, opb, opc, opd, answer_idx, task="medmcqa"}

Rules:
- Cache to cache_dir. Set HF_HOME to CHPC scratch on cluster.
- max_samples: shuffle then truncate, seed=42.
- Raises ValueError on empty dataset or schema mismatch; RuntimeError on HuggingFace
  load failure. Only the CLI entry point catches these and prints readable stderr + exit 1.
  Benchmark loaders must fail loudly — silent [] returns mask data problems.

---

### src/data/formatter.py

Responsibility: convert raw QA dicts to instruction-tuning strings using the
model's tokenizer chat template.

Public interface:
```python
def format_example(
    item: dict,
    tokenizer,
    add_generation_prompt: bool = False,  # False=train, True=eval
) -> str

def format_dataset(
    items: list[dict],
    tokenizer,
    add_generation_prompt: bool = False,
) -> list[str]
```

PubMedQA user message:
```
Context: {item['context']}

Question: {item['question']}

Answer with exactly one word: yes, no, or maybe.
```
Assistant message (training only): {item['answer']}

MedMCQA user message:
```
Question: {item['question']}
A) {item['opa']}
B) {item['opb']}
C) {item['opc']}
D) {item['opd']}

Answer with exactly one letter: A, B, C, or D.
```
Assistant message (training only): {answer_letter}

The system prompt is empty string for both tasks — the base chat template's
default system framing is used. Do not inject custom system prompts as this
can cause distribution mismatch between training and eval.

---

### src/train/lora_config.py

Responsibility: build PEFT LoraConfig and BitsAndBytesConfig from a loaded
config YAML dict. No side effects — pure config construction.

```python
def build_lora_config(cfg: dict) -> peft.LoraConfig
def build_bnb_config(cfg: dict) -> transformers.BitsAndBytesConfig
```

---

### src/train/train.py

Entry point for training. Reads config YAML via --config flag.

```
python src/train/train.py --config config/gpu_config.yaml [--dry-run]
```

--dry-run: load model + tokenizer + 10 samples, run one forward pass,
print batch shape and loss, then exit. Does not write any checkpoint.
Use this for local validation before submitting SLURM job.

Training loop:
1. Load config
2. Load and format datasets (call loader + formatter)
3. Run contamination check: remove pqa_artificial samples whose pubid overlaps
   with pqa_labeled eval set (src/data/contamination_check.py). Log N removed.
4. Build 5% internal validation split from combined training data (seed=42).
   This split is for early stopping only — NOT the PubMedQA or MedMCQA eval sets.
5. Build LoRA + BnB configs
6. Load base model with 4-bit quantization
7. Call prepare_model_for_kbit_training(model) — REQUIRED before applying LoRA
   when using bitsandbytes 4-bit. Enables gradient checkpointing and upcasting
   of layer norms to float32. Missing this call causes silently wrong gradients.
8. Apply LoRA adapter (peft.get_peft_model)
9. Use trl.SFTTrainer with DataCollatorForCompletionOnlyLM for assistant-only
   loss masking. The response template marks the start of the assistant turn.
   Prompt tokens (can be hundreds for PubMedQA context) are excluded from loss.
   Set packing=False — DataCollatorForCompletionOnlyLM requires this; packing
   breaks the response template boundary detection and silently removes masking.
   Dry-run assertion: after formatting 10 samples, verify that the label tensor
   contains -100 for all prompt tokens and valid token IDs only for answer tokens.
   Derive response_template token IDs from the actual rendered Llama chat template
   (apply_chat_template output), not from a hardcoded string.
10. Truncation: if formatted example exceeds max_seq_length, truncate from the
    LEFT (remove beginning of context), preserving question + answer tokens.
11. Save adapter to outputs/checkpoints/{run_name}/
12. Save training_log.json (loss curve, final val loss) to outputs/logs/

Output artifacts:
- outputs/checkpoints/{run_name}/adapter_config.json
- outputs/checkpoints/{run_name}/adapter_model.safetensors
- outputs/logs/{run_name}/training_log.json

run_name: {model_nickname}_{YYYYMMDD_HHMMSS}

---

### src/convert/merge_lora.py

Responsibility: merge LoRA adapter into base model weights, save as BF16.
Does not quantize — produces full-precision merged weights for GGUF input.

```
python src/convert/merge_lora.py \
    --config config/gpu_config.yaml \
    --adapter outputs/checkpoints/{run_name}/ \
    --output outputs/merged/{run_name}/
```

Uses: peft.PeftModel.merge_and_unload() → save_pretrained(safe_serialization=True)
Output: HuggingFace model directory with full BF16 weights (~16GB for 8B).

---

### src/convert/to_gguf.py

Responsibility: convert a HuggingFace model directory to GGUF, then quantize
at a specified level. Wraps llama.cpp's conversion and quantization binaries.

```
python src/convert/to_gguf.py \
    --model-dir outputs/merged/{run_name}/ \
    --output-dir outputs/gguf/{run_name}/ \
    --levels Q2_K Q4_K_M Q6_K Q8_0 F16 \
    --llama-cpp-dir /path/to/llama.cpp/
```

Steps per level:
1. python llama.cpp/convert_hf_to_gguf.py --model-dir ... --outtype f16 → model-f16.gguf
   Pass --outtype f16 explicitly even though current llama.cpp defaults to f16,
   because defaults can change across llama.cpp commits and the result manifest
   must make the precision choice auditable without re-reading conversion logs.
2. llama.cpp/llama-quantize model-f16.gguf model-{level}.gguf {level}

F16 conversion only runs step 1 (no quantization step needed).

Base model GGUF: the base model (downloaded BF16 weights) also goes through
step 1 with --outtype f16 to produce the base F16 GGUF reference. This ensures
all 10 variants are evaluated through the same inference code path.
Each level is run independently — one failure does not block others.
Logs each step output to stderr.

CHPC note: llama.cpp must be cloned and built on CHPC before this script
runs. See docs/CHPC-setup.md for build instructions.

---

### src/eval/eval.py

Responsibility: run inference on a single model variant for a single dataset.
Returns an EvalResult with accuracy, latency stats, and model metadata.

```python
@dataclass
class EvalResult:
    model_path: str
    model_variant: str              # e.g. "finetuned_q4_k_m"
    is_finetuned: bool
    quantization: str               # F16 | Q8_0 | Q6_K | Q4_K_M | Q2_K
    dataset: str                    # pubmedqa | medmcqa
    n_samples: int
    accuracy_abs: float             # raw accuracy on full eval set
    drop_from_f16: float            # F16_acc - Qx_acc for same model family
    brittleness_delta: float        # finetuned_drop - base_drop at same level
                                    # Positive = fine-tuned more brittle
    confidence_interval_95: tuple[float, float]  # Wilson 95% CI on accuracy_abs [lower, upper]
    drop_ci_95: tuple[float, float]              # paired bootstrap 95% CI on drop_from_f16
    brittleness_delta_ci_95: tuple[float, float] # paired bootstrap 95% CI on brittleness_delta
    invalid_output_rate: float      # fraction of samples where logprob failed
    model_size_gb: float
    ttft_p50_ms: float              # time-to-first-token p50, latency pass only
    ttft_p95_ms: float              # time-to-first-token p95, latency pass only
    tps_mean: float                 # tokens per second (mean over 100 latency prompts)
    hardware: str                   # A800 | M1-Air
    eval_backend: str               # "llama_cpp" — provenance field
    prompt_template_version: str    # "v1" — provenance field; bump if template changes
    run_id: str                     # training checkpoint name, e.g. "llama-3.1-8b_20260501_143200"
                                    # ties eval records to a specific training run; metadata only
    timestamp: str                  # ISO-8601 audit field, not a dedup key

def eval_model(
    model_path: str,
    model_variant: str,
    dataset: list[dict],
    config: dict,
    n_warmup: int = 50,
) -> EvalResult
```

SINGLE BACKEND: All 10 model variants are evaluated through llama.cpp via
llama-cpp-python. There is no HuggingFace Transformers backend for eval.
All variants — including the F16 base and F16 fine-tuned references — are
loaded as GGUF files. This guarantees identical tokenization, chat template
application, and stop-token handling across all variants.

Answer extraction — logprob scoring (not greedy decode):
1. Build the full formatted prompt (including instruction + question + context)
   with add_generation_prompt=True to append the assistant turn marker.
2. Tokenize each candidate answer separately:
   - PubMedQA: ["yes", "no", "maybe"]
   - MedMCQA: ["A", "B", "C", "D"]
3. Call llama_cpp.Llama with logprobs=True. Extract the log probability of
   each candidate token at the first answer position.
4. Take argmax — the candidate with highest log probability is the prediction.
   This is deterministic and robust to generation instability at Q2.

Label tokenization policy: bare vs leading-space labels (e.g. "A" vs " A")
is determined by verify/check_logprob_scoring.py in Session 4.0 using the
actual Llama-3.1-8B tokenizer. The chosen form must produce single-token
coverage for all candidates. If any label is multi-token, use sum of token
logprobs (teacher forcing). Record the chosen form in metrics/versions.lock.

If logprob API fails for a variant: record invalid_output_rate=1.0 and
exclude that variant from brittleness_delta analysis.

---

### src/eval/batch_eval.py

Responsibility: orchestrate all 10 eval runs, collect results, write
metrics/results.json. This is the main eval entry point.

```
python src/eval/batch_eval.py \
    --config config/gpu_config.yaml \
    --base-gguf-dir outputs/gguf/base/ \
    --finetuned-gguf-dir outputs/gguf/{run_name}/ \
    --raw-output-dir metrics/raw/
```

Each invocation writes results for ONE model variant to:
  metrics/raw/{model_variant}.json

No file locking is needed — one file per job. This is safe on CHPC VAST storage
where fcntl is unreliable on network filesystems.

Skips variants whose raw file already exists (idempotent re-run).
Prints progress table to stdout as each variant completes.

Parallel eval on CHPC: slurm/launch_eval.sh submits one eval.sbatch job per
model variant (10 jobs total). Each job writes its own raw file independently.

After all 10 jobs complete, merge raw files into results.json:
```
python src/eval/merge_results.py \
    --raw-dir metrics/raw/ \
    --output metrics/results.json
```

merge_results.py validates all 10 raw files present, checks dedup key uniqueness
(model_variant + dataset + quantization + hardware), computes drop_from_f16 and
brittleness_delta across the combined records, and writes metrics/results.json.

---

### src/analysis/charts.py

Responsibility: read metrics/results.json, produce two charts.

Chart 1 — Quality Cliff:
  X-axis: quantization level (Q2_K → F16)
  Y-axis: accuracy
  Two lines: base model + fine-tuned model
  One chart per dataset (2 charts total)
  Annotate the delta at each level

Chart 2 — Efficiency Frontier:
  X-axis: model size (GB)
  Y-axis: accuracy
  Each point = one model variant, labeled
  Separate markers for base vs fine-tuned

Output: metrics/charts/quality_cliff_{dataset}.png,
        metrics/charts/efficiency_frontier_{dataset}.png

---

### src/serve/app.py

FastAPI app serving the Q4_K_M fine-tuned model via llama-cpp-python.
Runs locally on Mac M1 Air with Metal acceleration.

Endpoint:
  POST /answer
  Body: {"question": str, "context": str | null, "task": "pubmedqa" | "medmcqa",
         "options": {"a": str, "b": str, "c": str, "d": str} | null}
  Response: {"answer": str, "raw_output": str, "latency_ms": int}

Model loaded once at startup from path specified in env var GGUF_MODEL_PATH.
No in-memory state between requests.

---

## CONFIG SCHEMA

Both gpu_config.yaml and mac_config.yaml share the same schema.
Fields marked [GPU only] are read when framework=trl and ignored when framework=mlx.

```yaml
# Framework
framework: str                  # "trl" (GPU/A800 QLoRA) | "mlx" (Mac LoRA fallback)

# Model
model_id: str                   # HuggingFace model ID
model_nickname: str             # short name used in file paths + result labels

# LoRA
lora_r: int
lora_alpha: int
lora_dropout: float
lora_target_modules: list[str]

# BitsAndBytes / QLoRA — [GPU only, ignored when framework=mlx]
quantization_bits: int          # 4
bnb_4bit_compute_dtype: str     # bfloat16 | float16
bnb_4bit_quant_type: str        # nf4

# Training
batch_size: int
gradient_accumulation_steps: int
max_seq_length: int
learning_rate: float
num_train_epochs: int
lr_scheduler: str               # cosine
warmup_ratio: float
max_samples_per_dataset: int | null  # null = full dataset; cap per dataset before mix

# Eval / Reproducibility
prompt_template_version: str    # "v1" — bump if prompt format changes

# Paths (CHPC scratch or local)
scratch_base: str               # base path for all outputs
hf_cache_dir: str               # HF_HOME path
llama_cpp_dir: str              # full path to llama.cpp build on CHPC
base_model_dir: str             # path to downloaded base HF weights
```

---

## RESULTS SCHEMA

metrics/results.json is a list of EvalResult objects:

```json
[
  {
    "model_variant": "finetuned_q4_k_m",
    "is_finetuned": true,
    "quantization": "Q4_K_M",
    "base_model": "llama-3.1-8b",
    "dataset": "pubmedqa",
    "n_samples": 1000,
    "accuracy_abs": 0.742,
    "drop_from_f16": 0.031,
    "brittleness_delta": 0.018,
    "confidence_interval_95": [0.714, 0.769],
    "drop_ci_95": [0.018, 0.044],
    "brittleness_delta_ci_95": [0.005, 0.031],
    "invalid_output_rate": 0.0,
    "model_size_gb": 4.7,
    "ttft_p50_ms": 145,
    "ttft_p95_ms": 220,
    "tps_mean": 42.3,
    "hardware": "A800",
    "eval_backend": "llama_cpp",
    "prompt_template_version": "v1",
    "run_id": "llama-3.1-8b_20260501_143200",
    "timestamp": "2026-05-01T14:32:00Z"
  },
  ...
]
```

One record per (model_variant × dataset) combination.
10 model variants × 2 datasets = 20 records in a complete run.

Dedup key: model_variant + dataset + quantization + hardware.
eval_backend and prompt_template_version are provenance fields — recorded for
reproducibility but not part of the dedup key (both are constant in this study).
timestamp is an audit field only.

drop_from_f16 and brittleness_delta are computed by merge_results.py after all
20 raw records exist. They require the F16 baseline for the same model family and
cannot be computed per-record during eval.

---

## SLURM JOB TEMPLATES

### slurm/train.sbatch
```bash
#!/bin/bash
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --job-name=medquant-train
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shreyas.patiledu07@gmail.com

module load cuda/12.1 python/3.11   # verify exact names with: module spider cuda
source ~/envs/medquant/bin/activate
export HF_HOME=/scratch/general/vast/u1527145/medquant/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /uufs/chpc.utah.edu/common/home/u1527145/projects/medquant
python src/train/train.py --config config/gpu_config.yaml
```

### slurm/eval.sbatch
Same header as train.sbatch.
Command: python src/eval/batch_eval.py --config config/gpu_config.yaml [args]

### slurm/convert.sbatch
Same header as train.sbatch (can also run on CPU partition — conversion is CPU-bound).
Command: python src/convert/merge_lora.py [args] && python src/convert/to_gguf.py [args]

---

