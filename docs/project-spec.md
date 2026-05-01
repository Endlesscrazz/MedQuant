# MedQuant — Project Specification
# Written: 2026-04-27
# Purpose: problem statement, research question, datasets, eval design,
#          success criteria, and interview narrative.
# ──────────────────────────────────────────────────────────────────────

## THE PROBLEM

Deploying LLMs in clinical settings requires two things that appear to be in
tension: domain accuracy (fine-tuning) and memory efficiency (quantization).
Practitioners routinely do both — fine-tune a general model on medical data,
then quantize it with GGUF for edge deployment where patient data cannot leave
the device.

The question nobody has answered cleanly: does fine-tuning on domain-specific
data make the model more sensitive to quantization? If a fine-tuned model
degrades faster under aggressive quantization than its base counterpart, then
practitioners are implicitly choosing between quality and efficiency, without
knowing where the breakpoint is.

---

## RESEARCH QUESTION

**Primary:** Does QLoRA fine-tuning on medical domain data increase a general
LLM's sensitivity to GGUF quantization compared to the base model?

**Secondary:** At which quantization level (Q2/Q4/Q6/Q8/F16) does the
fine-tuning quality gain become negligible due to quantization degradation?
Is there a "sweet spot" where fine-tuned model quality exceeds base at the
same memory budget?

---

## HYPOTHESIS

Fine-tuned models gain accuracy at full precision (F16/BF16) because
QLoRA adapts specific weight distributions to the medical domain. These
adapted distributions may have lower entropy tolerance — they encode more
task-specific information in fewer weights, making them more sensitive to
per-weight approximation error introduced by quantization.

Predicted outcome:
- Fine-tuned model outperforms base at F16/Q8 by a meaningful margin (≥5%).
- At Q2/Q4, the brittleness_delta is positive — fine-tuned model degrades
  faster because adapted weights encode more task-specific structure.
- Q4_K_M may prove to be a practical deployment point, but the experiment
  will identify the actual accuracy/memory frontier empirically.

This hypothesis may be wrong. Both directions are interesting findings.

---

## WHAT MAKES THIS NOVEL

Existing literature:
- Fine-tuning papers measure base vs fine-tuned accuracy at full precision.
- Quantization papers measure base model quality across bit-widths.
- Few papers compare quantization sensitivity of base vs fine-tuned models on
  the same downstream task with the same quantization pipeline.

This project contributes a clean empirical benchmark for the medical QA domain.
The result is a practitioners' reference: "For medical QA deployment, here is
where fine-tuning pays off and here is where quantization erases the gain."

NOTE: We do not claim this is the first study of LoRA + quantization interaction.
The novelty claim is narrower and more defensible: this is a controlled comparison
using a consistent inference backend (llama.cpp GGUF) across all variants, with
two complementary medical NLP benchmarks, and with brittleness_delta as the
primary reported metric. That combination is the contribution.

Connection to MED-MIR (prior work): MED-MIR used ONNX INT8 quantization for
BioMedCLIP inference efficiency. MedQuant asks the complementary question for
LLMs: does the fine-tuned model tolerate quantization the same way?

---

## DATASETS

### PubMedQA
- Source: qiaojin/PubMedQA on HuggingFace. No credentialing required.
- Task: Given a biomedical research question + abstract, answer yes/no/maybe.
- Size: 1,000 expert-labeled (eval) + 211,269 artificially generated (train).
- Splits used:
  - Train: pqa_artificial subset, split="train" (211K)
  - Eval: pqa_labeled subset, split="train" (1K expert-labeled)
  CRITICAL: pqa_labeled does NOT have a "test" split. Calling split="test"
  raises KeyError. All 1K expert-labeled samples are under split="train".
  See docs/dataset.md for complete loading patterns.
- Input format:
  ```
  Context: {abstract}
  Question: {question}
  Answer with exactly one word: yes, no, or maybe.
  ```
- Evaluation: 3-class accuracy on pqa_labeled (n=1,000)
- Baseline (literature): GPT-4 ~78%, Llama-3.1-8B zero-shot ~65%
- Contamination control: before training, remove pqa_artificial samples whose
  pubid appears in pqa_labeled. Expected overlap: 0, but check is mandatory.
  See src/data/contamination_check.py.

### MedMCQA
- Source: openlifescienceai/medmcqa on HuggingFace. No credentialing required.
- Task: USMLE-style multiple choice (4 options) covering clinical medicine.
- Size: ~182,822 train (after deduplication) / 4,183 validation / 6,150 test.
- Splits used:
  - Train: train split (~182K after filter to choice_type="single")
  - Eval: validation split (4,183 with known labels)
  NOTE: The test split labels are withheld by dataset authors. Validation split
  is the standard held-out eval in MedMCQA literature. Do not call it "test set."
  See docs/dataset.md for reporting label conventions.
- Filter: choice_type == "single" before both training and eval. Multi-select
  questions cannot be evaluated reliably with single-label logprob scoring.
- Input format:
  ```
  Question: {question}
  A) {opa}
  B) {opb}
  C) {opc}
  D) {opd}
  Answer with exactly one letter: A, B, C, or D.
  ```
- Evaluation: 4-class accuracy on validation split (n=4,183)
- Baseline (literature): GPT-4 ~87%, Llama-3.1-8B zero-shot ~60%

### Training dataset construction
Default: use full datasets. PubMedQA artificial (211K post-contamination-check)
+ MedMCQA train (~182K post-filter) = ~393K samples.
If a 1K-sample timed smoke job on A800 projects training > 5 hours for 2 epochs,
apply max_samples_per_dataset cap in gpu_config.yaml:
  - GPU cap: 150K per dataset → 300K total
  - Mac fallback (MLX): 50K per dataset → 100K total
Subsampling applied independently per dataset with seed=42 before combining.

All training data formatted as instruction-tuning pairs using the model's
chat template (apply_chat_template from the tokenizer). No system prompt
injection — let the base chat template handle system framing.

---

## MODEL CONFIGURATIONS

### Primary — Llama-3.1-8B-Instruct (GPU profile)
HuggingFace ID: meta-llama/Llama-3.1-8B-Instruct
Requires HuggingFace access token (Meta gated model).
Download size: ~16GB (BF16 weights)

QLoRA configuration:
- Quantization: 4-bit NF4 (bitsandbytes)
- LoRA rank (r): 16
- LoRA alpha: 32 (effective scaling = alpha/r = 2.0)
- LoRA dropout: 0.05
- Target modules: q_proj, v_proj, k_proj, o_proj (attention only)
- Training dtype: bfloat16

Training configuration:
- Batch size: 32, gradient accumulation: 1 (effective batch: 32)
- Max sequence length: 2048 tokens
- Learning rate: 2e-4, cosine schedule, warmup ratio 0.03
- Epochs: 2 (with early stopping on validation loss)
- Optimizer: paged_adamw_8bit (from bitsandbytes)
- Gradient checkpointing: enabled

### Fallback — Qwen-2.5-1.5B-Instruct (Mac profile)
HuggingFace ID: Qwen/Qwen2.5-1.5B-Instruct
No gating — open download.
Download size: ~3GB (BF16)

Training framework: mlx-lm (Apple MLX). NOT bitsandbytes QLoRA.
bitsandbytes does not reliably support Apple Silicon MPS backend and
will fail silently or crash on M1 Air. MLX is the correct local framework.

MLX fine-tuning configuration:
- Method: LoRA (mlx-lm built-in, --train flag)
- LoRA rank: 8, alpha: 16
- Target modules: q_proj, v_proj
- Quantization: 4-bit via --quantize flag in mlx_lm.convert

Training configuration:
- Batch size: 4, gradient accumulation: 8 (effective batch: 32)
- Max sequence length: 1024 tokens
- Same learning rate, scheduler as GPU profile

Note: Mac fallback produces a smaller model with different fine-tuning
quality. It answers the same quantization sensitivity question but at
reduced scale. Report results separately with "Mac M1 Air / MLX" hardware label.

---

## QUANTIZATION LEVELS

Five GGUF quantization levels per model variant (base + fine-tuned = 10 total):

| Level | Description | Expected size (8B) | Notes |
|---|---|---|---|
| F16 | Full half-precision | ~16GB | Reference quality ceiling |
| Q8_0 | 8-bit scalar quantization | ~8.5GB | Near-lossless baseline |
| Q6_K | 6-bit k-quant | ~6.1GB | High quality, moderate compression |
| Q4_K_M | 4-bit k-quant medium | ~4.7GB | Standard deployment target |
| Q2_K | 2-bit k-quant | ~2.7GB | Aggressive compression, quality floor |

K-quants (Q4_K_M, Q6_K, Q2_K) use importance-weighted quantization per block.
They generally outperform scalar quants (Q4_0, Q8_0) at the same bit-width.
Q4_K_M is chosen over Q4_0 as the deployment target because it preserves more
quality for the same storage cost.

Note on F16: all F16 eval artifacts are GGUF files produced by the same
convert_hf_to_gguf.py command — never raw HuggingFace weights.
- Base F16:     downloaded base HF weights (BF16) → convert_hf_to_gguf.py --outtype f16
- Fine-tuned F16: merged LoRA weights (BF16)     → convert_hf_to_gguf.py --outtype f16
Both paths produce the same file format and are evaluated through identical
llama.cpp code. This ensures tokenization is consistent across all 10 variants.

---

## EVAL METRICS

Five metrics recorded per (model_variant × dataset) record:

### 1. accuracy_abs — Task Accuracy (context metric)
- PubMedQA: 3-class accuracy (yes/no/maybe) on pqa_labeled (n=1,000)
- MedMCQA: 4-class accuracy on validation split (n=4,183)
- Method: logprob scoring via llama.cpp (NOT greedy decode).
  Compute log probability of each candidate token at the answer position,
  conditioned on the full prompt. Take argmax. Deterministic, no sampling.
  Robust to first-token fragility at aggressive quantization levels (Q2).
- All 10 variants evaluated through the same llama.cpp GGUF backend.

### 2. drop_from_f16 — Quantization Degradation (primary per-family metric)
- drop_from_f16 = accuracy_abs[F16] - accuracy_abs[Qx] for same model family.
- Computed independently for base model and fine-tuned model.
- Shows how much quality each model family loses at each compression level.

### 3. brittleness_delta — The Research Finding (central metric)
- brittleness_delta = finetuned_drop_from_f16 - base_drop_from_f16 at same Qx.
- Positive: fine-tuned model degrades more (more brittle under quantization).
- Zero: same sensitivity to quantization.
- Negative: fine-tuned model is more robust.
- This is the answer to the primary research question.

### 4. confidence_interval_95
- Wilson 95% confidence interval on accuracy_abs.
- With n=1,000 (PubMedQA): ±~3% at 70% accuracy.
- With n=4,183 (MedMCQA): ±~1.5% at 70% accuracy.
- Reported as [lower, upper] in results.json.

### 5. invalid_output_rate
- Fraction of samples where logprob scoring returned no valid result.
- Expected 0.0 for all variants. Non-zero signals GGUF corruption or API issue.
- Variants with invalid_output_rate > 0 are excluded from brittleness_delta.

### Model Size on Disk
- Reported as GB of the GGUF file.
- Directly represents the edge deployment storage cost.

### Inference Latency
- Measure: time-to-first-token (TTFT), p50 and p95 across eval set.
- Methodology: 50 warmup prompts, then measure on full eval set.
- Hardware recorded in every result. Latency numbers are never compared across hardware.

---

## EXPERIMENT DESIGN

The study has 2 × 5 = 10 conditions (model variant × quantization level).

```
                  Q2_K   Q4_K_M   Q6_K   Q8_0   F16
Base model         [1]    [2]      [3]    [4]    [5]
Fine-tuned model   [6]    [7]      [8]    [9]    [10]
```

Primary analysis: for each quantization level and dataset, compute drop_from_f16
separately for base and fine-tuned models, then compute:
  brittleness_delta = finetuned_drop_from_f16 - base_drop_from_f16
A positive and growing brittleness_delta across lower bit-widths confirms the
hypothesis. Plot brittleness_delta vs. quantization level.

Raw fine-tuned-minus-base accuracy delta is secondary context only — it
conflates fine-tuning gain with quantization sensitivity and must not be the
headline finding.

Secondary analysis: efficiency frontier plot — accuracy vs. model size.
Which model variant gives the best accuracy for a given GB budget?

---

## SUCCESS CRITERIA

Minimum bar (project is valid and reportable):
- [ ] All 10 model variants evaluated on both datasets
- [ ] Results table populated: 10 rows × 5 columns (2 datasets + 3 metrics)
- [ ] At least one clear quantitative finding, even if the hypothesis is wrong
- [ ] FastAPI endpoint serving the Q4_K_M fine-tuned model (local, Mac)

Strong bar (interview-worthy):
- [ ] Fine-tuned model beats base on PubMedQA by ≥ 5% at F16/Q8
- [ ] Clear inflection point identified (the quantization level where gain reverses)
- [ ] Efficiency frontier identifies a deployment sweet spot, whether or not it is Q4_K_M
- [ ] README with honest numbers that can be cited in interviews

Stretch bar (publishable direction):
- [ ] Repeat for two model sizes (8B + 1.5B) and compare whether sensitivity
      scales with model size
- [ ] Ablation: LoRA rank effect on quantization sensitivity (r=8 vs r=16 vs r=32)

---

## NON-GOALS

1. Clinical deployment or safety evaluation. This is a research benchmark, not
   a production medical AI system. No clinical decisions should rely on it.
2. RLHF or preference optimization. Standard SFT (QLoRA) only.
3. Multi-modal input. Text-only QA tasks. (BioMedCLIP's domain is separate.)
4. Beating state-of-the-art benchmarks. This is a comparative study, not a
   leaderboard submission.
5. HuggingFace model deployment or public API. Local inference only in v1.
6. Automated feedback learning or online adaptation. Eval is offline, static.

---

## INTERVIEW NARRATIVE

"In my MED-MIR project I quantized BioMedCLIP with ONNX INT8 for serverless
deployment — that taught me quantization is a first-class concern in medical AI.
In MedQuant I took that a step further: I wanted to know whether fine-tuning
a general LLM on medical data and then quantizing it for edge deployment gives
you the best of both worlds, or whether fine-tuning makes the model more fragile
under compression. I fine-tuned Llama-3.1-8B with QLoRA on PubMedQA and
MedMCQA, then ran all ten model variants — five quantization levels for both
base and fine-tuned — through a single llama.cpp inference backend using logprob
scoring, and measured where the fine-tuning gains collapse. The key metric is
brittleness_delta: the difference in accuracy drop between the fine-tuned and
base model at each quantization level, which directly answers whether the
domain-adapted weights are more fragile to compression than the general ones."

(Fill in the specific finding — what brittleness_delta showed — once results
are in. Do not pre-claim the Q4_K_M sweet spot before running the experiment.)
