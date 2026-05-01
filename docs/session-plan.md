# MedQuant — Session Task Tracker
# Written: 2026-04-27 | Last revised: 2026-04-30
# Purpose: session-by-session implementation plans with task checklists.
#          The authoritative reference for what to build next and what is done.
# ──────────────────────────────────────────────────────────────────────────

## HOW TO USE THIS DOCUMENT

At session start: find the first incomplete session. Verify prerequisites
  are met before writing any code.

During build: check off tasks as you complete them. Do not batch.

After session: mark session COMPLETE. Update CURRENT STATE in CLAUDE.md.
  Then update verify/MEDQUANT-SESSION-CHECKLIST.md with the CB ground truth.

Rule: never begin a session without a written plan. Never declare done
  with unchecked boxes or failing tests.

---

## SESSION STATUS OVERVIEW

| Session | Goal | Status | Gate |
|---|---|---|---|
| 1 | Repo setup, data pipeline, CB instrumentation | 🔲 PENDING | Setup |
| 2 | QLoRA training script + SLURM job submission | 🔲 PENDING | Train |
| 3 | LoRA merge + GGUF conversion at 5 levels | 🔲 PENDING | Convert |
| 4 | Eval harness + logprob spike + full results table | 🔲 PENDING | Eval |
| 5 | Analysis charts + FastAPI serve + wrap-up | 🔲 PENDING | Ship |

---

## SESSION 1 — Repo setup, data pipeline, CB instrumentation

Status: 🔲 PENDING

Goal: project skeleton in place, both datasets load cleanly and are verified
on CHPC, CB instrumentation is set up, reproducibility lock written.

### Prerequisites
- [ ] Meta Llama-3.1-8B license accepted at huggingface.co/meta-llama/Llama-3.1-8B-Instruct
      (gated — approval can take hours. Do this BEFORE Session 1 starts.)
- [ ] HuggingFace access token created and saved on CHPC login node
- [ ] CHPC environment set up per docs/CHPC-setup.md (Steps 1–4 complete)

Note: git repo initialization is task 1.0, not a prerequisite.

---

### 1.0 — Repo initialization
- [ ] `git init`, create .gitignore, make initial commit
- [ ] .gitignore must include:
      outputs/, *.gguf, *.safetensors, *.bin, *.pt, *.pth,
      hf_cache/, __pycache__/, .env, metrics/raw/
- [ ] pyproject.toml with dev dependencies:
      pytest, transformers, datasets, peft, bitsandbytes, trl, accelerate,
      llama-cpp-python, fastapi, uvicorn, pyyaml, matplotlib, scipy
      (mlx-lm listed as optional/Mac-only comment)
- [ ] Create all directories:
      src/common/, src/data/, src/train/, src/eval/, src/convert/,
      src/analysis/, src/serve/, slurm/, tests/, verify/ground_truth/,
      metrics/raw/, metrics/charts/
      Add .gitkeep to metrics/raw/ and metrics/charts/ so they track in git
- [ ] Add __init__.py to all src/ subdirectories
- [ ] Push to GitHub

### 1.1 — Config files
- [ ] Write config/gpu_config.yaml with full schema per architecture.md:
      framework: trl, model_id, model_nickname, LoRA params, BnB params,
      training params, max_samples_per_dataset: null, prompt_template_version: v1,
      scratch paths, llama_cpp_dir (placeholder — fill in Session 3),
      base_model_dir (set after CHPC download)
- [ ] Write config/mac_config.yaml:
      framework: mlx, Qwen-2.5-1.5B-Instruct, smaller LoRA/batch params,
      max_samples_per_dataset: 50000, same eval/path fields
- [ ] Verify both files parse cleanly with PyYAML

### 1.2 — Data pipeline (src/data/)
- [ ] Write src/data/loader.py
  - [ ] load_dataset() for pubmedqa:
        train: load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
        eval:  load_dataset("qiaojin/PubMedQA", "pqa_labeled",    split="train")
        CRITICAL: pqa_labeled has NO "test" split — calling split="test" raises KeyError.
  - [ ] load_dataset() for medmcqa (train + validation splits)
        Filter to choice_type == "single" before returning (drop multi-select rows)
  - [ ] max_samples supported: shuffle with seed=42, then truncate
  - [ ] TRANSFORMERS_OFFLINE, HF_DATASETS_OFFLINE, HF_HUB_OFFLINE supported
  - [ ] Returns standardized dict schema per architecture.md
  - [ ] Raises ValueError on empty dataset or schema mismatch; RuntimeError on
        load failure. Only CLI entry points catch and print readable stderr + exit 1.
  - [ ] Logs dataset size to stderr on load

- [ ] Write src/data/contamination_check.py
  - [ ] check_pubmedqa_contamination(train_items, eval_items) → (cleaned, n_removed)
        Primary key: pubid. Removes pqa_artificial items whose pubid appears in
        pqa_labeled eval set. Expected n_removed=0 but check is mandatory.
  - [ ] Log to stderr: "PubMedQA contamination check: removed N samples"
  - [ ] check_medmcqa_overlap(train_items, val_items) → int
        Normalized question text hash check. Expected 0. Log result.

- [ ] Write src/data/formatter.py
  - [ ] format_example(item, tokenizer, add_generation_prompt=False) → str
        Calls tokenizer.apply_chat_template() for both task types
  - [ ] PubMedQA user message: "Context: {context}\n\nQuestion: {question}\n\n
        Answer with exactly one word: yes, no, or maybe."
        Assistant message (train only): {answer}
  - [ ] MedMCQA user message: "Question: {question}\nA) ...\nB) ...\nC) ...\nD) ...\n
        \nAnswer with exactly one letter: A, B, C, or D."
        Assistant message (train only): {answer_letter}
  - [ ] add_generation_prompt=False for training, True for eval
  - [ ] format_dataset(items, tokenizer, add_generation_prompt) → list[str]

### 1.3 — Tests for data pipeline
- [ ] tests/test_data.py
  - [ ] test_pubmedqa_loader: mock HF, verify standardized schema
  - [ ] test_medmcqa_loader: mock HF, verify standardized schema, choice_type filter
  - [ ] test_formatter_pubmedqa: verify output contains context + question
  - [ ] test_formatter_medmcqa: verify output contains question + options
  - [ ] test_add_generation_prompt_false: no trailing assistant turn marker in output
  - [ ] test_max_samples: truncation respects max_samples with seed=42
  - [ ] test_loader_raises_on_empty: ValueError raised when dataset is empty
  - [ ] test_pubmedqa_contamination: PMID-matching removal logic
  - [ ] test_medmcqa_overlap: normalized hash check logic
- [ ] pytest tests/ — all tests green before proceeding

### 1.4 — CB instrumentation
- [ ] Create verify/MEDQUANT-SESSION-CHECKLIST.md with ground truth schema:
      ```json
      {
        "session": 1,
        "date": "YYYY-MM-DD",
        "files_touched": [],
        "artifacts_created": [],
        "chpc_job_ids": [],
        "scratch_paths": [],
        "key_decision": "",
        "last_completed": "",
        "next_step": "",
        "blockers": [],
        "continuity_level_expected": "assistant_derived"
      }
      ```
- [ ] Write verify/eval_cb_packet.py — reads JSONL from context-bridge MCP and
      scores a simulated session summary against ground truth schema above
- [ ] Verify /bridge command works at session start (loads prior session context)
- [ ] Add context-bridge SessionEnd hook: runs after each session to capture
      files_touched, artifacts_created, key decisions
- [ ] Fill in ground truth for Session 1 at end of 1.5

### 1.5 — CHPC setup verification + reproducibility lock
- [ ] SSH to CHPC, complete docs/CHPC-setup.md Steps 5–8
- [ ] Download Llama-3.1-8B-Instruct to scratch (or confirm already downloaded)
- [ ] Download both datasets to HF_HOME cache
- [ ] Verify PubMedQA:
      pqa_artificial len ~211K, pqa_labeled len exactly 1,000
- [ ] Verify MedMCQA:
      train len (after choice_type filter), validation len exactly 4,183
      Record actual filtered count — do not assume 182,822 without verifying
- [ ] Run contamination checks: log results to stderr
- [ ] Run quick smoke test on login node:
      python src/data/loader.py --config config/gpu_config.yaml
      (should print sizes, not crash)
- [ ] Write metrics/versions.lock with all fields from docs/dataset.md
      Reproducibility Record schema. Fill in real values, not placeholders.
      Note: llama_cpp_commit filled in Session 3; model_revision filled now.
- [ ] Commit versions.lock — never modify after this point

### 1.6 — Session wrap-up
- [ ] Update CLAUDE.md CURRENT STATE block
- [ ] Update verify/MEDQUANT-SESSION-CHECKLIST.md ground truth for Session 1
- [ ] Mark this session COMPLETE in this file

---

## SESSION 2 — QLoRA training script + SLURM job

Status: 🔲 PENDING

Prerequisites:
- [ ] Session 1 COMPLETE (all boxes checked, versions.lock committed)
- [ ] Llama-3.1-8B fully downloaded to CHPC scratch
- [ ] Datasets verified on CHPC (sizes match expected)
- [ ] pytest green (all Session 1 tests pass)

Goal: training script written and dry-run passes, SLURM job submitted and
running on CHPC, training log validated.

---

### 2.1 — LoRA config builder (src/train/lora_config.py)
- [ ] build_lora_config(cfg: dict) → peft.LoraConfig
      Reads lora_r, lora_alpha, lora_dropout, lora_target_modules from cfg
- [ ] build_bnb_config(cfg: dict) → BitsAndBytesConfig (GPU/TRL only)
      Returns None if cfg['framework'] == 'mlx'
- [ ] Both functions: no hardcoded values — read only from cfg dict

### 2.2 — Training script (src/train/train.py)
- [ ] Arguments: --config (yaml path), --dry-run (flag)
- [ ] Read framework from config: branch between TRL (GPU) and mlx_lm (Mac)
- [ ] TRL path (framework=trl):
  - [ ] Load tokenizer + base model with 4-bit NF4 quantization
  - [ ] REQUIRED: call prepare_model_for_kbit_training(model) BEFORE get_peft_model
        Missing this causes silently wrong gradients through quantized layers
  - [ ] Apply LoRA adapter via peft.get_peft_model()
  - [ ] Load + format training datasets (call loader + formatter)
  - [ ] Run contamination check (pubmedqa PMID + medmcqa text hash)
        Log counts removed to stderr
  - [ ] Create 5% internal validation split from combined training data:
        - After contamination removal, before training
        - seed=42, stratified by task (keep ~5% from each task)
        - This split is ONLY for early stopping; never used for reported metrics
  - [ ] Configure SFTTrainer with DataCollatorForCompletionOnlyLM:
        - packing=False (required — packing breaks response template detection)
        - response_template derived from actual chat template rendering, not hardcoded
        - Derive response template token IDs from tokenizer.apply_chat_template output
  - [ ] Dry-run assertion (always run, not just --dry-run):
        Format 10 samples → check label tensor: prompt tokens must be -100,
        answer tokens must be valid IDs. Raise AssertionError if masking fails.
  - [ ] Early stopping: eval_steps=500, save_steps=500,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        EarlyStoppingCallback(early_stopping_patience=3)
  - [ ] Left-truncation: if example > max_seq_length, remove from left
        (preserves question + answer at end of sequence)
  - [ ] Trainer args all read from config (no hardcoded values)
  - [ ] Save adapter to outputs/checkpoints/{run_name}/
  - [ ] Save training_log.json to outputs/logs/{run_name}/
        run_name = {model_nickname}_{YYYYMMDD_HHMMSS} — this becomes run_id in eval
- [ ] MLX path (framework=mlx):
  - [ ] Branch to mlx_lm.train() with equivalent config fields
  - [ ] Same dry-run, contamination check, data split logic
- [ ] --dry-run: load 10 samples, run 1 forward pass, print batch shape + loss,
      exit without saving. Verify masking assertion passes.

### 2.3 — SLURM job script (slurm/train.sbatch)
- [ ] Header per architecture.md SLURM template (mail, partition, nodes)
- [ ] module load commands — use exact verified names from module spider, not placeholders
- [ ] venv activate
- [ ] export HF_HOME, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1, HF_HUB_OFFLINE=1
- [ ] cd + python train.py command

### 2.4 — Local dry-run + CHPC smoke job
- [ ] GPU dry-run on CHPC login node:
      python src/train/train.py --config config/gpu_config.yaml --dry-run
      Expected: batch shape, forward pass loss (finite, not NaN), masking assertion
      passes, script exits cleanly
- [ ] Mac (MLX) dry-run locally:
      python src/train/train.py --config config/mac_config.yaml --dry-run
- [ ] Smoke job on CHPC: submit with max_samples_per_dataset=1000, num_train_epochs=1
      Measure wall time for 1K samples × 1 epoch
      If projected 2-epoch full-dataset time > 5 hours: set max_samples_per_dataset: 150000

### 2.5 — CHPC training submission
- [ ] git push all changes
- [ ] SSH to CHPC: git pull, sbatch slurm/train.sbatch
- [ ] Monitor: squeue -u u1527145
- [ ] After job completes: rsync training_log.json to local metrics/
- [ ] Inspect loss curve — train loss decreasing, val loss reasonable, no NaN

### 2.6 — Session wrap-up
- [ ] Update CLAUDE.md CURRENT STATE (training status: "adapter saved at ...")
- [ ] Update verify/MEDQUANT-SESSION-CHECKLIST.md ground truth for Session 2
      (include CHPC job ID, checkpoint scratch path, training run_id)
- [ ] Mark this session COMPLETE

---

## SESSION 3 — LoRA merge + GGUF conversion

Status: 🔲 PENDING

Prerequisites:
- [ ] Session 2 COMPLETE
- [ ] LoRA adapter checkpoint confirmed on CHPC scratch
- [ ] Training log shows convergence (loss decreased, no NaN)
- [ ] pytest still green

Goal: LoRA adapter merged into full BF16 weights, all 5 GGUF levels generated
for both base and fine-tuned model, all 10 GGUF files load cleanly.

---

### 3.1 — LoRA merge script (src/convert/merge_lora.py)
- [ ] Arguments: --config, --adapter (checkpoint path), --output
- [ ] Load base model in full precision (NOT 4-bit quantized for merge)
- [ ] peft.PeftModel.merge_and_unload()
- [ ] save_pretrained(output_dir, safe_serialization=True)
- [ ] Print merged model size to stdout
- [ ] Do NOT run on Mac — needs 32GB+ RAM for 8B. Run as SLURM job on CHPC.

### 3.2 — GGUF conversion script (src/convert/to_gguf.py)
- [ ] Arguments: --model-dir, --output-dir, --levels (list), --llama-cpp-dir,
      --model-type (base | finetuned)
- [ ] Step 1: python convert_hf_to_gguf.py --outtype f16 --outfile {name}-f16.gguf
      --outtype f16 passed ALWAYS (explicit = auditable even if default is f16)
- [ ] Step 2: for each level in [Q2_K, Q4_K_M, Q6_K, Q8_0]:
      llama-quantize {name}-f16.gguf {name}-{level}.gguf {level}
- [ ] F16 skips step 2 (step 1 output IS the F16 artifact)
- [ ] One failure does not block remaining levels
- [ ] Print file sizes of all generated GGUF files
- [ ] Write size_manifest.json to output-dir (model_variant, path, size_gb, sha256)
- [ ] Record llama_cpp_commit (git rev-parse HEAD) — add to metrics/versions.lock

### 3.3 — llama.cpp setup on CHPC (one-time, manual step)
- [ ] SSH to CHPC login node
- [ ] git clone https://github.com/ggerganov/llama.cpp to home dir
- [ ] module load cmake gcc + correct cuda module (from module spider output)
- [ ] cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j8
- [ ] Test: ./build/bin/llama-cli --version && ./build/bin/llama-quantize --help
- [ ] Record llama.cpp path; update llama_cpp_dir in config/gpu_config.yaml
- [ ] Record llama_cpp_commit in metrics/versions.lock (git rev-parse HEAD)

### 3.4 — SLURM conversion job (slurm/convert.sbatch)
- [ ] Merge step: needs 32GB RAM, CPU partition acceptable
- [ ] Conversion step: CPU-intensive, no GPU needed
- [ ] Single sbatch runs both steps sequentially

### 3.5 — Submit conversion jobs on CHPC
- [ ] Submit merge job for fine-tuned model
- [ ] Submit GGUF conversion for fine-tuned model (5 levels)
- [ ] Submit GGUF conversion for base model (5 levels — no merge, just convert HF BF16 weights)
      IMPORTANT: use the SAME base model weights used for LoRA training for base GGUF
- [ ] Verify all 10 GGUF files exist with expected approximate sizes:
      F16 ~16GB, Q8 ~8.5GB, Q6 ~6.1GB, Q4_K_M ~4.7GB, Q2_K ~2.7GB

### 3.6 — Smoke test + GPU offload verification
- [ ] For each GGUF: run one forward pass with llama-cli
      ./llama.cpp/build/bin/llama-cli -m model.gguf -p "Is ibuprofen an NSAID?" -n 5
      Confirm all 10 produce non-empty output (correctness not checked here)
- [ ] GPU offload verification (requires interactive GPU node via salloc):
      python -c "
      from llama_cpp import Llama
      m = Llama('/path/to/smallest.gguf', n_gpu_layers=-1, verbose=True)
      print(m('Hello', max_tokens=3))
      "
      Inspect verbose output for CUDA/cuBLAS confirmation.
      CPU-only is a silent error that distorts all latency measurements.
- [ ] Record size_manifest.json — commit to git (small file, no weights)

### 3.7 — Session wrap-up
- [ ] Fill in llama_cpp_commit in metrics/versions.lock; commit the update
- [ ] Update CLAUDE.md CURRENT STATE (10 GGUF files confirmed)
- [ ] Update verify/MEDQUANT-SESSION-CHECKLIST.md for Session 3
      (include scratch paths for all 10 GGUF files, conversion job IDs)
- [ ] Mark this session COMPLETE

---

## SESSION 4 — Eval harness + full results table

Status: 🔲 PENDING

Prerequisites:
- [ ] Session 3 COMPLETE
- [ ] All 10 GGUF files confirmed loading (Session 3.6 smoke test passed)
- [ ] GPU offload verified (verbose CUDA output confirmed)
- [ ] pytest still green

Goal: eval harness written and tested, logprob scoring spike validates the API,
all 10 model variants evaluated on both datasets, metrics/results.json populated
with 20 records including bootstrap CIs.

---

### 4.0 — Logprob scoring implementation spike (required before full eval)

Do this before writing any eval code. Full eval is blocked until this spike passes.

- [ ] Load one GGUF on CHPC using an interactive GPU node (salloc)
      Use base_q4_k_m or similar — not the smallest, but not F16 either
- [ ] Test both bare and leading-space label forms:
      PubMedQA: ["yes","no","maybe"] vs [" yes"," no"," maybe"]
      MedMCQA:  ["A","B","C","D"]    vs [" A"," B"," C"," D"]
      Log which form produces single-token coverage with Llama-3.1-8B tokenizer
- [ ] Confirm the logprob scoring flow:
      Build prompt → pass to Llama with logprobs=True → extract candidate log probs
      → argmax. Verify this produces a valid answer for a test prompt.
- [ ] ALSO test with F16 GGUF + a long PubMedQA prompt (abstract-length context).
      The F16 model has the largest memory footprint. If logits_all=True is required,
      verify it does not OOM on A800 with a long context.
- [ ] If any label is multi-token: implement teacher-forcing sum of token logprobs.
      Document which approach is used (single-token argmax or multi-token sum).
- [ ] Write verify/check_logprob_scoring.py — self-contained spike script.
      Script must print: chosen label forms, token IDs, a sample prediction,
      and whether the API works on both small and large GGUFs.
- [ ] Record chosen label form and scoring method in metrics/versions.lock.
- [ ] Only proceed to 4.1 once this spike produces correct answers on a test prompt.

### 4.1 — Metrics module (src/eval/metrics.py)
- [ ] score_labels_pubmedqa(model, prompt: str) -> str | None
      Score ["yes","no","maybe"] (or leading-space form from spike). Returns argmax.
      Returns None if logprob API fails.
- [ ] score_labels_medmcqa(model, prompt: str) -> str | None
      Score ["A","B","C","D"] (or leading-space form). Returns argmax or None.
- [ ] compute_accuracy(predictions: list, labels: list) -> float
- [ ] compute_wilson_ci(n_correct: int, n_total: int) -> tuple[float, float]
      95% Wilson CI on accuracy_abs
- [ ] compute_paired_bootstrap_ci(
          scores_a: list[int], scores_b: list[int],
          n_bootstrap: int = 10000, seed: int = 42
      ) -> tuple[float, float]
      Paired bootstrap CI for a difference metric (drop_from_f16 or brittleness_delta)
      scores_a and scores_b are paired 0/1 vectors over the same eval examples.
- [ ] compute_drop_from_f16(results: list[EvalResult]) -> dict
      For each model family: drop_from_f16[quantization] = f16_acc - qx_acc
- [ ] compute_brittleness_delta(results: list[EvalResult]) -> dict
      For each quantization level: finetuned_drop - base_drop
- [ ] measure_latency(model, prompts: list[str]) -> (p50_ms, p95_ms, tps_mean)
      Latency pass: 100 fixed prompts per dataset per variant (greedy generation).
      TTFT p50/p95 (ms) + TPS mean (tokens per second).
      Run as a separate pass — do NOT mix with logprob accuracy pass.
- [ ] get_model_size_gb(model_path: str) -> float
- [ ] tests/test_metrics.py:
  - [ ] test label scoring returns argmax of candidate tokens
  - [ ] test None returned when logprob API fails
  - [ ] test Wilson CI against known reference values
  - [ ] test paired bootstrap CI on fixture data (known difference)
  - [ ] test drop_from_f16 and brittleness_delta on known fixture

### 4.2 — Single model eval runner (src/eval/eval.py)
- [ ] eval_model(model_path, model_variant, dataset, config) -> EvalResult

Two separate passes per variant — do NOT mix them:

Accuracy pass:
- [ ] Load via llama_cpp.Llama(model_path=..., n_gpu_layers=-1)
      All 10 variants use llama.cpp — no HuggingFace Transformers backend ever.
- [ ] Logprob scoring using the label form and method from Session 4.0 spike
- [ ] Record accuracy_abs, Wilson CI; invalid_output_rate
- [ ] Set drop_from_f16 and brittleness_delta to -1.0 sentinel — post-hoc in merge
- [ ] Set run_id from config (derived from training checkpoint name in versions.lock)
- [ ] Set eval_backend="llama_cpp", prompt_template_version from config

Latency pass (separate call):
- [ ] Greedy generation on 100 fixed prompts (first 100 from eval set, consistent
      across all variants)
- [ ] Record ttft_p50_ms, ttft_p95_ms, tps_mean
- [ ] Do NOT report logprob scoring latency as deployment latency

- [ ] Returns EvalResult dataclass per architecture.md schema

### 4.3 — Batch eval orchestrator (src/eval/batch_eval.py)
- [ ] Arguments: --config, --variant, --base-gguf-dir, --finetuned-gguf-dir,
      --raw-output-dir
- [ ] Evaluates ONE model variant on both datasets (2 EvalResult records)
- [ ] Writes results to metrics/raw/{model_variant}.json
      No locking needed — each SLURM job writes its own file
- [ ] Skips if metrics/raw/{model_variant}.json already exists (idempotent re-run)
- [ ] Prints progress to stdout

### 4.4 — Merge script (src/eval/merge_results.py)
- [ ] Arguments: --raw-dir metrics/raw/, --output metrics/results.json
- [ ] Validates all 10 expected variant files present; errors loudly if any missing
- [ ] Dedup check on model_variant + dataset + quantization + hardware
- [ ] Computes drop_from_f16 and brittleness_delta for all 20 records
- [ ] Computes paired bootstrap CIs for drop_from_f16 and brittleness_delta
      using per-example prediction vectors (must be stored in raw files)
- [ ] Writes metrics/results.json — the final complete results table
- [ ] Prints summary table: variant, dataset, accuracy_abs, drop_from_f16, brittleness_delta

Note: raw/*.json files must store per-example predictions (not just aggregate accuracy)
so bootstrap CI can be computed in merge_results.py. Store as "predictions": [0,1,1,...]

### 4.5 — SLURM eval scripts
- [ ] slurm/eval.sbatch — evaluates one variant on both datasets, writes metrics/raw/
      Reads VARIANT env var from sbatch --export
- [ ] slurm/launch_eval.sh — submits 10 eval.sbatch jobs, one per model variant
      Each writes its own raw file. No shared file or locking needed.
      Example: sbatch --export=VARIANT=finetuned_q4_k_m slurm/eval.sbatch

### 4.6 — Run full eval on CHPC
- [ ] Run slurm/launch_eval.sh (submits 10 parallel jobs)
- [ ] Monitor: squeue -u u1527145
- [ ] After all jobs complete: verify 10 files in metrics/raw/
- [ ] Run merge: python src/eval/merge_results.py --raw-dir metrics/raw/ \
                 --output metrics/results.json
- [ ] rsync metrics/ to local (raw/ + results.json)
- [ ] Verify 20 records in results.json (10 variants × 2 datasets)
- [ ] Verify drop_from_f16 and brittleness_delta populated (not -1.0 sentinel)
- [ ] Verify bootstrap CIs populated and non-zero
- [ ] Spot-check: finetuned_f16 accuracy > base_f16 by some margin (if not, training failed)
- [ ] Spot-check: invalid_output_rate = 0.0 for all variants
- [ ] Spot-check: large accuracy inversions (Q4 > Q8 by > 3%) — investigate before accepting
- [ ] Log any reruns to metrics/eval_reruns.md (see rerun policy in docs/dataset.md)

### 4.7 — Session wrap-up
- [ ] Update CLAUDE.md CURRENT STATE (results table populated)
- [ ] Update verify/MEDQUANT-SESSION-CHECKLIST.md for Session 4
      (include all 10 eval SLURM job IDs, confirm raw file paths)
- [ ] Mark this session COMPLETE

---

## SESSION 5 — Analysis charts + FastAPI serve + wrap-up

Status: 🔲 PENDING

Prerequisites:
- [ ] Session 4 COMPLETE
- [ ] metrics/results.json has all 20 records with CIs populated
- [ ] pytest still green

Goal: quality cliff charts generated, FastAPI endpoint serving locally,
README written with honest numbers, project wrapped up.

---

### 5.1 — Analysis charts (src/analysis/charts.py)
- [ ] quality_cliff(results: list[EvalResult], dataset: str) → saves PNG
      X-axis: quantization level (Q2_K → F16)
      Y-axis: accuracy_abs
      Two lines: base + fine-tuned
      Annotate brittleness_delta at each quantization level
      One chart per dataset (2 total)
- [ ] brittleness_chart(results: list[EvalResult]) → saves PNG
      X-axis: quantization level
      Y-axis: brittleness_delta with 95% bootstrap CI error bars
      This is the primary finding chart — headline for README and interviews
- [ ] efficiency_frontier(results: list[EvalResult], dataset: str) → saves PNG
      X-axis: model_size_gb, Y-axis: accuracy_abs
      Labeled points, separate markers for base vs fine-tuned
- [ ] Generate all charts: 2 quality cliff + 1 brittleness + 2 frontier = 5 PNGs
- [ ] Save to metrics/charts/

### 5.2 — FastAPI serve endpoint (src/serve/app.py)
- [ ] Loads GGUF_MODEL_PATH from env var at startup
- [ ] POST /answer endpoint per architecture.md spec
- [ ] Response includes: answer, raw_output, latency_ms
- [ ] tests/test_serve.py:
  - [ ] test /answer returns valid JSON
  - [ ] test pubmedqa and medmcqa request schemas
  - [ ] test invalid task returns 422

### 5.3 — README.md (first and only time)
- [ ] Problem statement (2 sentences)
- [ ] Architecture diagram (ASCII, from architecture.md overview)
- [ ] Key findings table with actual numbers (brittleness_delta per quantization level)
- [ ] Brittleness chart embedded or linked
- [ ] Install + run instructions (local serve with GGUF)
- [ ] Honest labels: "Evaluated on A800 GPU, n=1,000 PubMedQA + n=4,183 MedMCQA"
- [ ] Reporting labels: "PubMedQA pqa_labeled eval (n=1,000)",
      "MedMCQA validation-as-held-out-eval (n=4,183)"
- [ ] Link to MED-MIR as related prior work

### 5.4 — Final verification
- [ ] All tests green: pytest tests/
- [ ] All 5 charts generated and committed
- [ ] README committed with real numbers
- [ ] Run `context-bridge simulate` to preview CB capture of this session
- [ ] Update verify/MEDQUANT-SESSION-CHECKLIST.md for Session 5

### 5.5 — Project wrap-up
- [ ] CLAUDE.md CURRENT STATE updated to "ALL 5 SESSIONS COMPLETE"
- [ ] DECISIONS.md has entries for all non-trivial implementation choices made
- [ ] metrics/versions.lock is complete and committed (all fields filled)
- [ ] .gitignore confirmed: no weights/GGUF/checkpoints accidentally committed
- [ ] Mark this session COMPLETE in this file
