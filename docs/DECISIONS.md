# MedQuant — Design Decision Log
# Written: 2026-04-27
# Purpose: append-only log of every non-trivial design decision.
#          Newest entries at top. Do not edit past entries.
#
# When to add an entry:
#   - A module's interface or schema changes
#   - A dataset preprocessing choice is made
#   - A training hyperparameter is selected with a specific reason
#   - A tool or library is chosen over an alternative
#   - Any choice that would surprise a future reader
#
# When NOT to add an entry:
#   - Bug fixes
#   - Minor config tweaks
#   - Test additions for existing code
#   - Renaming / formatting
# ──────────────────────────────────────────────────────────────────────────

## Entry template

### [Decision Name] — YYYY-MM-DD

**What was decided:**
[1–2 sentences]

**Why this approach:**
[Reasoning — 2–3 sentences]

**Alternatives considered:**
[What else was evaluated and why rejected]

**Trade-off accepted:**
[What was given up — be honest]

**How to explain in an interview:**
[3–5 sentences. Plain English. Problem → decision → result.]

---

## Pre-implementation decisions (resolved before Session 1)

---

### Single inference backend for all eval variants — 2026-04-28

**What was decided:**
All 10 model variants (base + fine-tuned × 5 quantization levels) are evaluated
through a single backend: llama.cpp via llama-cpp-python. All models are loaded
as GGUF files, including the F16 base and F16 fine-tuned references.

**Why this approach:**
Mixing llama-cpp-python for GGUF variants and HuggingFace Transformers for BF16/F16
variants introduces backend differences in tokenization, chat template application,
stop-token handling, and BOS/EOS behavior. Those differences can skew accuracy
comparisons and make brittleness_delta uninterpretable — the measured difference
would reflect backend drift, not quantization sensitivity.

**Alternatives considered:**
HuggingFace Transformers for F16 variants (original plan). Rejected because the
eval harness inconsistency was flagged as a fatal flaw: results could not answer
the central research question.

**Trade-off accepted:**
F16 GGUF is not bit-identical to BF16 HuggingFace weights. The BF16→F16 conversion
introduces minimal rounding error (~0.1% accuracy impact), which is acceptable because
the goal is consistent comparison, not absolute maximum F16 accuracy.

**How to explain in an interview:**
"The key design decision was to run all ten model variants through a single inference
backend — llama.cpp — instead of using HuggingFace for the full-precision models and
llama.cpp only for quantized ones. Mixing backends introduces tokenization and
template differences that can masquerade as quantization effects. Running everything
through the same code path means any accuracy difference is attributable to
quantization, not backend variance."

---

### Primary metric: brittleness (degradation-from-F16) — 2026-04-28

**What was decided:**
The primary reported metric is brittleness_delta = finetuned_drop_from_f16 minus
base_drop_from_f16 at each quantization level. Raw accuracy (accuracy_abs) is
reported as context, not the headline finding.

**Why this approach:**
Comparing raw accuracy at each quantization level conflates fine-tuning gain with
quantization sensitivity. A fine-tuned model might show higher absolute accuracy at Q4
simply because it started from a higher F16 baseline, not because it is more robust.
Measuring degradation from each model's own F16 reference isolates the quantization
effect from the fine-tuning effect.

**Alternatives considered:**
Reporting only accuracy_abs at each quantization level (original plan). Rejected because
it does not isolate the quantization sensitivity question — the project's central finding.

**Trade-off accepted:**
drop_from_f16 and brittleness_delta cannot be computed until all 10 variants are
evaluated. They require the F16 baseline as a reference point, so partial results
during eval are not interpretable on their own.

---

### PubMedQA evaluation: all 1,000 expert-labeled samples — 2026-04-28

**What was decided:**
Use all 1,000 pqa_labeled samples as a single eval set. No 500/500 dev/test split.

**Why this approach:**
A dev/test split is only justified when you need a held-out set for prompt tuning,
threshold calibration, or hyperparameter selection on the eval task. MedQuant does none
of these — logprob scoring selects the argmax directly with no tunable threshold. Using
all 1,000 maximizes statistical power (Wilson 95% CI of ±~3% at 70% accuracy vs ±~4.4%
on 500 samples). The eval set is run exactly once per variant with no iterative peeking.

**Alternatives considered:**
500 dev / 500 final test. Rejected because there is no task that requires the dev split,
so the split halves statistical power for no gain.

**Trade-off accepted:**
We cannot report performance on the official 500-question PubMedQA leaderboard split.
Reports must use the label "PubMedQA pqa_labeled eval (n=1,000)" — not "test set."

---

### Mac fallback: MLX-LM on Qwen-2.5-1.5B — 2026-04-28

**What was decided:**
If CHPC access is lost, train with mlx-lm on Qwen-2.5-1.5B-Instruct on Mac M1 Air.
bitsandbytes QLoRA on Apple Silicon MPS is explicitly ruled out.

**Why this approach:**
bitsandbytes does not reliably support Apple Silicon MPS. It either fails silently or
crashes. mlx-lm is the correct framework for LoRA fine-tuning on Apple Silicon — it uses
MLX (Apple's ML framework) with full Metal GPU acceleration.

**Alternatives considered:**
bitsandbytes QLoRA on MPS (original plan). Rejected because it is not a reliable path
and would produce silently incorrect training runs. CPU-only training was also considered
but is too slow for any meaningful dataset size.

**Trade-off accepted:**
Qwen-2.5-1.5B is a smaller model than Llama-3.1-8B. Mac results are a reduced-scale
study — the brittleness_delta question is still answerable but results are not directly
comparable to the GPU run. Mac results must be labeled "Mac M1 Air / MLX / Qwen-2.5-1.5B"
and reported separately.

---

### Parallel eval: per-variant raw files + merge script — 2026-04-28

**What was decided:**
Each SLURM eval job writes its results to metrics/raw/{model_variant}.json independently.
After all 10 jobs complete, src/eval/merge_results.py validates and merges the raw files
into metrics/results.json.

**Why this approach:**
The alternative — 10 jobs concurrently appending to a shared results.json via fcntl.flock
— is fragile on CHPC VAST (a network filesystem). fcntl locking semantics are not
guaranteed on NFS/VAST mounts. A lock failure causes silent data corruption in results.json,
which is difficult to detect and impossible to recover without re-running all jobs.
One file per job requires zero coordination: no locking, no race conditions.

**Alternatives considered:**
fcntl.flock concurrent writes to shared results.json. Rejected because of network
filesystem reliability concerns on CHPC VAST.

**Trade-off accepted:**
Requires a separate merge step after all jobs complete. The merge script must handle
partial completions (some raw files present, others not) and report which variants are
missing before computing cross-variant metrics.

---

### Training task mixture: natural ratio, no resampling — 2026-04-28

**What was decided:**
PubMedQA artificial (~211K) and MedMCQA train (~182K) are concatenated and shuffled
with seed=42 without resampling. The natural ratio (~1.16:1) is close enough that
task-balanced sampling adds implementation complexity for negligible benefit.

**Why this approach:**
Task-balanced sampling is only necessary when datasets differ dramatically in size.
At 1.16:1, both tasks receive substantial representation in every training epoch.
The risk of format overfitting (model learns PubMedQA 3-class format at the expense
of MedMCQA 4-class format) is mitigated by reporting per-dataset eval results
separately — if one task degrades, it is detectable and reportable as a finding.

**Alternatives considered:**
1:1 task-balanced sampling (equal samples from each dataset per batch). Rejected as
over-engineered at this dataset size ratio. Separate training runs per dataset. Rejected
because the project tests mixed medical QA fine-tuning, not task-specific fine-tuning.

**Trade-off accepted:**
PubMedQA has slightly more influence on training loss due to its larger size. This is
acceptable given the close ratio and the per-dataset eval reporting.

---

### Answer extraction: logprob scoring primary, regex fallback — 2026-04-28

**What was decided:**
Primary answer extraction uses direct log probability scoring of candidate labels
(["yes","no","maybe"] or ["A","B","C","D"]) at the answer position via llama-cpp-python.
Greedy text generation is not used for accuracy measurement.

**Why this approach:**
Greedy first-token extraction has two failure modes: (1) the model may produce whitespace,
BOS/EOS artifacts, or an explanation before the answer letter — especially common at Q2
where generation is unstable; (2) different quantization levels may place the answer token
at different positions, making first-token extraction inconsistent. Logprob scoring
sidesteps both by measuring the model's internal belief directly.

**Alternatives considered:**
Greedy decode + regex extraction (original plan). Rejected as the primary method due to
fragility at aggressive quantization levels. Kept as fallback for cases where the logprob
API is unavailable.

**Trade-off accepted:**
Logprob scoring requires a Session 4 implementation spike to verify the llama-cpp-python
API works correctly for this use case before running full eval. Latency cannot be measured
from the logprob scoring pass — latency requires a separate generation run.

---

## Decisions logged during implementation sessions below this line
