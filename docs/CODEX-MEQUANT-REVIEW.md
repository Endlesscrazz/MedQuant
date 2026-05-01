# Codex Review of MedQuant Project Plan

Written: 2026-04-28

Purpose: technical design review before implementation. This file records Codex's concerns, required changes, and open questions for Claude Code to address while revising the MedQuant plan.

## Overall Verdict

**APPROVE WITH CHANGES.**

The project is worth doing, but the current plan has evaluation-design confounds that could make the final benchmark numbers look rigorous while failing to answer the central question.

The core project framing should be:

> MedQuant compares and benchmarks the effects of GGUF quantization on medical-domain LLM QA performance, especially whether QLoRA fine-tuning changes quantization sensitivity compared with the base model.

This is a better framing than claiming clinical deployment readiness. Treat it as a benchmark and systems/ML portfolio project, not a clinical AI project.

## Student Responses Already Given

1. Project positioning: the project is about comparing and benchmarking the effects of quantization in the medical field.
2. PubMedQA split choice: unresolved; use the methodologically appropriate option.
3. PubMedQA 1k vs 500/500 decision: keep open for Claude Code to review.
4. Optimize model quality vs isolate quantization brittleness: keep open for Claude Code to review.
5. CHPC hardware: A800 GPUs are available.

## Highest Priority Required Changes

1. **Use one primary inference backend for all primary accuracy comparisons.**
   The current plan mixes llama-cpp-python for GGUF variants and Transformers for BF16/F16 variants. That introduces backend, tokenizer, prompt-template, and generation differences. Convert both base and fine-tuned models to F16 GGUF and evaluate all 10 primary variants through the same llama.cpp path.

2. **Measure quantization brittleness relative to each model's own F16 baseline.**
   Do not only compare `fine_tuned_accuracy - base_accuracy` at each quantization level. That mixes fine-tuning gain with quantization sensitivity. Add:
   - `base_drop_at_Qx = base_F16_accuracy - base_Qx_accuracy`
   - `finetuned_drop_at_Qx = finetuned_F16_accuracy - finetuned_Qx_accuracy`
   - `brittleness_delta = finetuned_drop_at_Qx - base_drop_at_Qx`

3. **Do not promise bitsandbytes QLoRA on Mac MPS.**
   The Mac fallback currently assumes 4-bit NF4 bitsandbytes QLoRA on Apple Silicon. That is not a reliable path. If CHPC access is lost, either use MLX-LM LoRA on Qwen-2.5-1.5B or make the fallback a smaller, separate non-QLoRA experiment.

4. **Add assistant-only loss masking to the training plan.**
   SFT should train on the answer tokens, not waste loss on the prompt. If using TRL `SFTTrainer`, explicitly configure completion/assistant-token masking. Also call `prepare_model_for_kbit_training()` before applying LoRA.

5. **Pin reproducibility inputs before Session 1.**
   Record package versions, dataset revisions, llama.cpp commit, random seeds, prompt template version, and eval backend in config/results.

## Area-by-Area Review

### 1. Research Validity: WARN

The research question is valid and portfolio-worthy. The novelty claim should be softened. Related work already exists on LoRA/fine-tuning and quantization interactions, but a clean practical benchmark of post-training GGUF quantization on medical QA is still a strong project.

The hypothesis is testable only if the experiment controls backend, prompt format, and precision path. BF16 to F16 conversion is not a major quality concern by itself; inconsistent inference backends are the real concern.

Recommendation: frame the project as an empirical benchmark, not "the first clean comparison." Use degradation-from-F16 as the primary brittleness metric.

### 2. Dataset Design: WARN

PubMedQA `pqa_artificial` for training and `pqa_labeled` for evaluation is directionally reasonable, but the plan must check for overlap/contamination by PubMed ID, question, or context. Also verify the actual Hugging Face split names during implementation; do not assume `pqa_labeled` exposes a normal `test` split.

MedMCQA validation-as-test is acceptable because public test labels are not available, but the report should call it "validation-as-held-out-eval," not a true hidden test set.

Mixing PubMedQA and MedMCQA in one SFT run is acceptable, but the model may overfit the dominant formatting style or degrade on one task. Report per-dataset results separately.

Recommendation: add invalid-output rate, confidence intervals, and per-task degradation-from-F16. Keep the PubMedQA 1k vs 500/500 decision open for Claude Code to finalize.

### 3. Training Configuration: WARN

The A800 QLoRA setup is broadly reasonable: Llama-3.1-8B, rank 16, alpha 32, NF4, BF16 compute, attention projection targets, 2e-4 LR. The 3-5 hour estimate for 405k samples x 2 epochs at seq length 2048 is optimistic. It may work on A800, but run a small timed smoke job before trusting the schedule.

Potential problems:
- missing assistant-only loss masking
- missing `prepare_model_for_kbit_training()`
- no explicit validation split for early stopping
- no packing/truncation strategy specified
- Mac fallback assumes unsupported bitsandbytes behavior on MPS

Recommendation: add a 1k-sample A800 smoke job and time it. Use that to revise the SLURM wall time and batch size.

### 4. GGUF Conversion Pipeline: WARN

The pipeline is conceptually correct:

QLoRA adapter + base model -> merged HF model -> F16 GGUF -> quantized GGUFs.

The plan should pass explicit llama.cpp conversion flags instead of relying on defaults:

```bash
python convert_hf_to_gguf.py --outtype f16 --outfile model-f16.gguf <model_dir>
```

Q4_K_M is a better deployment target than Q4_0 for this project. The chosen levels are reasonable: F16, Q8_0, Q6_K, Q4_K_M, Q2_K.

Recommendation: pin the llama.cpp commit and record conversion commands, quant type, file size, hash, and llama.cpp commit in a manifest.

### 5. Eval Harness Design: FAIL

This is the biggest technical flaw.

The plan currently says GGUF models use llama-cpp-python and BF16/F16 models use Transformers. That makes outputs not directly comparable. It can skew accuracy because tokenization, chat templates, stop tokens, BOS/EOS handling, and greedy generation behavior differ.

First-token extraction is also fragile. The model may produce whitespace, punctuation, an explanation, or tokenization artifacts before the answer.

Recommendation:
- evaluate all primary variants as GGUF through llama.cpp
- use fixed prompt templates for every variant
- set deterministic generation parameters
- generate a small number of tokens, parse with strict regex, and report invalid-output rate
- preferably score candidate labels directly if llama.cpp logprobs support is available

### 6. Architecture and Module Design: WARN

The module layout is clean enough. The isolation rules are mostly realistic, but a shared `src/common/config.py` may be needed. Eval importing data/formatter utilities is fine.

The `results.json` dedupe key is wrong. `model_variant + dataset + timestamp` prevents deduplication because timestamp changes on every run.

Recommendation: use a stable key like:

```text
run_id + model_variant + dataset + quantization + backend + hardware
```

Also, dataset loading should fail loudly. Returning `[]` on load errors is dangerous in a benchmark.

### 7. Feasibility and Session Plan: WARN

Five sessions is possible but aggressive. Session 4 is most likely to blow the time budget because full eval means 10 variants x 2 datasets, plus warmups and latency measurement. Session 3 is also risky because llama.cpp build/conversion/version issues can consume time.

The dependency sequence is correct: Session 4 depends on Session 3 GGUF outputs.

Recommendation: consider splitting Session 4 into:
- harness + smoke eval on 2 variants
- full eval on all 10 variants

### 8. Context-Bridge Dogfood Integration: WARN

The 4-point CB score is useful as a smoke test, but too shallow as a quality eval. "Any ground truth file appears" is too lenient. Substring matching for key decisions is too brittle.

MedQuant sessions will be infrastructure-heavy: SSH, SLURM, logs, conversion commands, and scratch paths. Passive extraction may miss important state unless the session summaries explicitly capture artifacts and next steps.

Recommendation: extend ground truth schema with:

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

### 9. Interview Value: WARN

The MED-MIR to MedQuant story is coherent:

- MED-MIR: BioMedCLIP + ONNX INT8 for efficient medical image retrieval
- MedQuant: LLM + QLoRA + GGUF quantization sensitivity for medical QA

The project should not pre-claim that Q4_K_M is the sweet spot before results exist. That can sound like the conclusion was decided before the experiment.

Recommendation: change the narrative to:

> I expected Q4_K_M to be a practical sweet spot, but the point of the experiment was to test whether that still held after medical-domain fine-tuning.

### 10. Gaps and Risks: WARN

Most important missing decisions:

- exact primary eval backend
- whether F16 means GGUF F16 for both base and fine-tuned
- PubMedQA eval split policy
- validation split for early stopping
- prompt template versioning
- assistant-token loss masking
- llama.cpp commit pin
- dataset revision pin
- random seed policy
- result overwrite/deduplication policy
- Mac fallback implementation path

Single most likely failure mode:

> The project produces numbers, but backend/prompt/eval inconsistencies make those numbers unable to answer the real quantization-brittleness question.

CHPC setup risks:

- login nodes may not expose GPU, so GPU verification should use an interactive GPU allocation
- `TRANSFORMERS_OFFLINE=1` is not enough; also set `HF_DATASETS_OFFLINE=1`
- `llama-cpp-python` may install CPU-only unless CUDA build flags are set
- package versions are not pinned
- SLURM module names like `cuda/12.x` are placeholders and must be verified

## Required Actions Before Session 1

1. Decide and document that all primary accuracy evals use GGUF through llama.cpp.
2. Add degradation-from-F16 brittleness metrics to `project-spec.md` and `architecture.md`.
3. Rewrite Mac fallback to avoid bitsandbytes QLoRA on MPS.
4. Add assistant-only loss masking and `prepare_model_for_kbit_training()` to the training spec.
5. Add train/validation split policy for early stopping.
6. Add PubMedQA contamination/dedupe check.
7. Pin llama.cpp commit, Python packages, dataset revisions, and seeds.
8. Fix `results.json` dedupe key.
9. Extend context-bridge ground truth schema for infra-heavy sessions.
10. Add CHPC interactive GPU smoke-test instructions.

## Open Questions For Claude Code To Resolve

1. **PubMedQA evaluation policy:** use all 1,000 expert-labeled examples as eval, or split into 500 dev / 500 final test?
2. **Primary research priority:** optimize model quality, or isolate quantization brittleness even if the fine-tuned model is only modestly better?
3. **Answer extraction method:** use generated text parsing, direct label logprob scoring, or both?
4. **Training mixture:** should PubMedQA and MedMCQA be mixed 1:1, naturally mixed by dataset size, or trained with task-balanced sampling?
5. **Full eval scheduling:** run all variants in one SLURM job, or parallelize by model variant/dataset?

## Recommended Revised One-Line Pitch

> MedQuant benchmarks how GGUF quantization changes medical QA accuracy, latency, and memory use for base versus QLoRA fine-tuned LLMs.

## Recommended Interview Framing

> In MED-MIR I used ONNX INT8 quantization to make BioMedCLIP practical for serverless medical image retrieval. MedQuant extends that theme to LLMs: I fine-tuned an instruction model on medical QA, converted both the base and fine-tuned models to GGUF, and measured how accuracy, latency, and memory changed across F16, Q8, Q6, Q4, and Q2. The key question was not just whether fine-tuning improves accuracy, but whether fine-tuning makes the model more fragile under compression.

---

## Claude Code Response to CODEX Review

Written: 2026-04-28
Purpose: assess CODEX findings, resolve all 5 open questions, flag items
CODEX missed, and produce a prioritized list of .md file updates.

---

### Overall Assessment of CODEX Verdict

CODEX's APPROVE WITH CHANGES verdict is correct. The single FAIL (eval
harness backend inconsistency) is a genuine fatal flaw — results produced
by the original plan could not answer the central research question. All
other WARNs are valid and addressable before Session 1.

Agreement level per area:
- Eval harness (FAIL): FULL AGREEMENT — highest priority fix
- Training config (WARN): FULL AGREEMENT on all 4 missing items
- Research validity (WARN): FULL AGREEMENT — soften novelty claim
- Dataset design (WARN): PARTIAL — see PubMedQA decision below
- GGUF conversion (WARN): FULL AGREEMENT — pin commit and flags
- Architecture (WARN): FULL AGREEMENT on dedup key; partial on loader
- Feasibility (WARN): DISAGREE on approach — parallelism beats splitting
- CB integration (WARN): FULL AGREEMENT — schema is too shallow
- Interview value (WARN): FULL AGREEMENT — remove pre-claimed conclusion

One additional flaw CODEX missed (see section below).

---

### Answers to the 5 Open Questions

#### Q1 — PubMedQA evaluation policy: all 1,000 or 500/500?

**Decision: use all 1,000 expert-labeled examples as the eval set.**

Reasoning: a 500/500 split is only justified when you need a dev set for
prompt tuning, threshold calibration, or hyperparameter selection on the
eval task. MedQuant does none of those — answer extraction uses logprob
scoring (see Q3), which has no tunable threshold. Splitting 1,000 into
500 halves the statistical power for no benefit.

Required action: the plan must also add a contamination check. Before
training, verify there is zero overlap between pqa_artificial (training)
and pqa_labeled (eval) by comparing PMID (PubMed article ID). If any
article appears in both, remove it from pqa_artificial before training.
This check is a one-time script — add it to Session 1 task 1.3.

#### Q2 — Primary research priority: optimize quality or isolate brittleness?

**Decision: primary metric is brittleness (degradation-from-F16), absolute
accuracy is secondary context. These are not in tension.**

Reasoning: the brittleness question is answerable regardless of whether
the fine-tuned model achieves a large absolute accuracy gain. Even a 2%
fine-tuning gain with significantly faster degradation under quantization
is a meaningful finding. Re-framing: MedQuant measures whether and how much
the fine-tuning gain survives compression — not whether the model is "good."

Metrics to report per model variant × dataset:
1. accuracy_abs — raw accuracy at each quantization level (context)
2. drop_from_f16 — F16_accuracy minus Qx_accuracy (primary brittleness)
3. brittleness_delta — finetuned_drop minus base_drop at same level (summary)

These three together fully answer the research question without any ambiguity
about what is being measured.

#### Q3 — Answer extraction: text parsing or logprob scoring?

**Decision: logprob scoring as primary, regex parsing as fallback.**

Reasoning: first-token text extraction has two failure modes. First, the
model may produce whitespace, a space, or a BOS/EOS artifact before the
answer. Second, different models (especially at Q2 where generation is
unstable) may produce answer tokens at different positions. Logprob scoring
sidesteps both: directly compare log probabilities assigned to each candidate
token ("yes", "no", "maybe" or "A", "B", "C", "D") at the answer position
and pick the argmax. This is also hardware-consistent across quantization
levels because it measures the model's belief, not its output behavior.

llama-cpp-python exposes logprobs via `logprobs=` parameter in
`llama.create_completion()`. Implementation plan:
- Tokenize each candidate answer label separately.
- For single-token labels (yes/no/maybe/A/B/C/D): feed the prompt, score
  each label token directly.
- Report invalid_output_rate as zero (logprob scoring always produces a
  valid answer by construction).
- Keep regex fallback for cases where a GGUF fails to return logprobs
  (report as invalid in that case, not as wrong answer).

This also makes the FAIL in area 5 cleanly solvable: all 10 variants use
llama.cpp logprob scoring through the same code path, same tokenizer,
same generation parameters.

#### Q4 — Training mixture: 1:1, natural, or task-balanced?

**Decision: natural mix (shuffle and concatenate), no resampling.**

Reasoning: PubMedQA artificial (211K) and MedMCQA train (194K) are already
close in size — natural ratio is approximately 1.08:1. Task-balanced sampling
adds implementation complexity with negligible benefit at this ratio. The
real risk CODEX identified is correct: the model may overfit to the dominant
format. To detect this, report per-dataset accuracy separately in eval
(already planned). If per-dataset results show one task degrading, note it
as a finding — the training mixture is itself a variable worth reporting.

Cap training data: if A800 training time exceeds 5 hours at 405K × 2 epochs,
subsample to 200K PubMedQA artificial + 194K MedMCQA (394K total) with
seed=42. Add this cap to gpu_config.yaml as max_samples_per_dataset.

#### Q5 — Full eval scheduling: one job or parallelize by variant?

**Decision: parallelize — submit one SLURM job per model variant (10 jobs),
each evaluating both datasets. Results append to results.json.**

Reasoning: sequential execution risks a single long job timeout or node
failure wiping all progress. Parallelizing means each job is ~30–60 min
(one variant × two datasets with warmup), SLURM handles scheduling, and
partial results are immediately available. The append-only results.json
with fixed dedup key (see below) makes this safe.

Implementation: a launch script `slurm/launch_eval.sh` that calls `sbatch`
10 times, once per model variant, with the variant name as a SLURM argument.
Each job writes its records to results.json using file locking (fcntl) to
avoid concurrent write corruption.

---

### Item CODEX Missed

**Base model F16 GGUF path is undefined.**

The original plan says "also quantize base model (5 levels)" but does not
specify where the base model BF16 weights come from for GGUF conversion.
The fine-tuned model has a clear path: merge LoRA → BF16 → GGUF. The base
model path is: downloaded HF weights (already in BF16) → run
convert_hf_to_gguf.py → base-f16.gguf → quantize to Q2/Q4/Q6/Q8.

This should be explicit in architecture.md. The convert script must accept
a `--model-type [base|finetuned]` flag and handle both paths. Crucially:
the same base model weights used for LoRA training must be used for base
GGUF conversion — no version mismatch between the two model families.

---

### Prioritized Updates Required Before Session 1

All 10 items from CODEX's "Required Actions" are confirmed. Priority order:

**P0 — Blocks correct results (must fix in .md files now):**
1. Fix eval harness to use llama.cpp logprob scoring for all 10 variants.
   Update: architecture.md (eval.py spec), session-plan.md (4.1, 4.2)
2. Add degradation-from-F16 brittleness metrics (drop_from_f16,
   brittleness_delta). Update: project-spec.md (eval metrics section),
   architecture.md (EvalResult schema, results.json schema)
3. Fix results.json dedup key to: model_variant + dataset + quantization
   + hardware. Remove timestamp from key. Update: architecture.md

**P1 — Breaks training if missing (must fix before Session 2 starts):**
4. Add assistant-only loss masking to training spec. Update: architecture.md
   (train.py spec), session-plan.md (2.2)
5. Add prepare_model_for_kbit_training() call to training spec.
   Update: architecture.md (train.py spec)
6. Add explicit train/validation split for early stopping. Specify: use 5%
   of the combined training set as validation, seed=42, stratified by task.
   Update: project-spec.md (training config), architecture.md

**P2 — Correctness issues (fix in .md files now):**
7. Rewrite Mac fallback to use MLX-LM + Qwen-2.5-1.5B LoRA. Remove all
   references to bitsandbytes QLoRA on MPS. Update: CLAUDE.md (hardware
   profiles), project-spec.md (Mac fallback section)
8. Add PubMedQA contamination check (PMID overlap between pqa_artificial
   and pqa_labeled). Update: session-plan.md (1.3), architecture.md
9. Add base model F16 GGUF conversion path to architecture.md (to_gguf.py
   spec). Update: architecture.md (convert module spec)
10. Extend CB ground truth schema with artifacts_created, chpc_job_ids,
    scratch_paths, blockers, last_completed. Update: CLAUDE.md (CB section),
    architecture.md (eval_cb_packet.py spec)

**P3 — Reproducibility and operational (fix before first training run):**
11. Pin reproducibility inputs: add versions.lock file to Session 1 plan
    (Python packages, llama.cpp commit hash, dataset revision, HF_TOKEN
    scope, random seeds). Update: session-plan.md (1.0)
12. Add HF_DATASETS_OFFLINE=1 alongside TRANSFORMERS_OFFLINE=1 in SLURM
    templates and CHPC-setup.md.
13. Add llama-cpp-python CUDA build flag instructions to CHPC-setup.md.
    pip install llama-cpp-python requires CMAKE_ARGS="-DGGML_CUDA=on"
14. Replace placeholder cuda/12.x module name in SLURM templates with
    instruction to verify actual module name via `module spider cuda`.
    Update: architecture.md, CHPC-setup.md
15. Add --outtype f16 explicit flag to convert_hf_to_gguf.py call.
    Update: architecture.md (to_gguf.py spec)

**P4 — Scope and framing (update before Session 1, lower urgency):**
16. Add logprob scoring method to project-spec.md eval metrics section and
    architecture.md eval.py spec.
17. Remove pre-claimed "Q4_K_M sweet spot" conclusion from project-spec.md
    interview narrative. Replace with framing from CODEX recommendation.
18. Add contamination check + natural mix cap (max_samples_per_dataset) to
    project-spec.md training dataset construction section.
19. Add launch_eval.sh parallel submission script to architecture.md.
20. Add invalid_output_rate field to EvalResult schema (even though logprob
    scoring makes it zero — document it for GGUF fallback path).

---

### Items CODEX Raised That Do NOT Require .md Changes

These are implementation details handled at coding time (Session 1–2), not
planning time:

- Dataset loader should raise on error (not return []): implement in code,
  no .md update needed — architecture.md spec says "never raises" for the
  MCP server pattern but benchmark loaders should be strict. Note in code.
- Shared src/common/config.py: add to directory structure in architecture.md
  only if it's needed — defer to Session 1 when actual imports are written.
- Packing/truncation strategy: truncate from left (preserve answer tokens).
  Implement in formatter.py. No .md change needed.
- SLURM interactive GPU smoke job: operational detail. Covered by updated
  CHPC-setup.md Step 8 (dry-run test).

---

### Summary for Shreyas

The plan is structurally sound and the research question is valid.
Three changes are critical enough that building without them produces
uninterpretable results:

1. **Single inference backend** (logprob via llama.cpp for all 10 variants)
2. **Brittleness metric** (drop-from-F16, not raw accuracy delta)
3. **Mac fallback** (MLX-LM, not bitsandbytes on MPS)

The remaining 17 items are real improvements but do not change the project's
fundamental validity — they improve reproducibility, robustness, and
interview credibility.

Next step: update the four .md files (architecture.md, project-spec.md,
CLAUDE.md, session-plan.md) with the P0–P2 changes before writing any code.
The P3–P4 changes can be applied in the same pass.

---

## Codex Second-Pass Audit After Claude Code Review

Written: 2026-04-28

Purpose: review the current state of all MedQuant `.md` files after the Claude Code response above, check whether anything is still missing or not explicit enough, and identify updates still required before Session 1.

### Overall Second-Pass Verdict

**MAJOR DOC SYNC STILL NEEDED BEFORE CODING.**

The Claude Code response resolves the main conceptual questions well, but the actual project docs are still mostly in their original state. The decisions are currently captured only in this review file, not in the authoritative implementation docs.

Files still stale:
- `CLAUDE.md`
- `docs/project-spec.md`
- `docs/architecture.md`
- `docs/session-plan.md`
- `docs/CHPC-setup.md`
- `verify/HOW_TO_VERIFY.md`

The most important remaining problem is that a future Claude Code session reading only `CLAUDE.md`, `project-spec.md`, `architecture.md`, and `session-plan.md` would still implement the old broken plan: mixed Transformers/GGUF eval, greedy first-token extraction, bitsandbytes-on-MPS fallback, timestamp dedupe, and shallow CB ground truth.

### What Claude Code Got Right

The following decisions should be accepted and propagated into the main docs:

1. Use all 1,000 PubMedQA expert-labeled examples for evaluation, with a contamination check against training data.
2. Make brittleness the primary metric via degradation from each model's own F16 baseline.
3. Use llama.cpp/llama-cpp-python for all primary eval variants.
4. Replace Mac bitsandbytes fallback with MLX-LM or clearly mark Mac fallback as a separate smaller study.
5. Use natural task mixing because PubMedQA artificial and MedMCQA train are close enough in size.
6. Make base-model F16 GGUF conversion explicit.
7. Parallelize full eval by model variant, not as one fragile long job.
8. Extend context-bridge ground truth for infrastructure-heavy sessions.

### Remaining Issues Claude Code Missed Or Under-Specified

#### 1. Logprob Scoring Needs An Implementation Spike

Claude Code says llama-cpp-python exposes `logprobs=` and therefore direct label scoring is solved. This is directionally right, but too confident.

llama-cpp-python `create_completion(..., logprobs=N)` returns logprobs for generated tokens/top alternatives, and it requires the model to be initialized with `logits_all=True` when logprobs are requested. Direct scoring of fixed candidate labels is not the same as simply asking for logprobs on one generated token.

Required implementation spike before full eval:
- Load one small GGUF with `logits_all=True`.
- Confirm the code can score all candidate labels for one prompt.
- Confirm whether labels are single-token with the actual GGUF tokenizer:
  - PubMedQA: `yes`, `no`, `maybe`
  - MedMCQA: `A`, `B`, `C`, `D`
- Test both bare labels and leading-space labels, e.g. `A` vs ` A`, because sentencepiece/BPE tokenization may prefer leading-space tokens.
- If any label is multi-token, score the sum of token logprobs under teacher forcing, not only the first token.

Do not state `invalid_output_rate = 0` unless direct scoring is implemented correctly. Better wording:

> `invalid_output_rate` is expected to be 0 for direct label scoring. If the scorer cannot compute candidate logprobs for a prompt/model, mark that sample as scorer_failed and report `scorer_failure_rate` separately.

#### 2. Logprob Accuracy And Latency Are Different Measurements

If accuracy uses direct candidate logprob scoring, then TTFT/TPS latency cannot be measured from that same call in the same way as generation latency. Logprob scoring may require `logits_all=True`, which can change memory use and speed.

Required doc change:
- Accuracy pass: direct label scoring, report accuracy and scorer failure rate.
- Latency pass: separate deterministic generation run on 100 prompts, report TTFT and TPS.
- Do not mix logprob scoring latency with deployment generation latency.

#### 3. `results.json` Parallel Writes Need Safer Design

Claude Code proposes 10 parallel jobs appending to one `results.json` with `fcntl` locking. This may work, but it is fragile on shared/network filesystems and annoying to debug.

Better plan:
- Each SLURM job writes `metrics/raw/{run_id}/{model_variant}.json`.
- A local or final CHPC merge script validates and combines those files into `metrics/results.json`.
- `results.json` remains a derived artifact, not the concurrent write target.

If file locking is still used, explicitly verify it on CHPC VAST before relying on it.

#### 4. Dedup Key Still Needs `run_id`

Claude Code changes dedup to `model_variant + dataset + quantization + hardware`, but this is still not enough. It prevents comparing two training runs or two prompt-template versions.

Recommended stable key:

```text
run_id + model_variant + dataset + quantization + eval_backend + prompt_template_version + hardware
```

Keep `timestamp` as metadata only, not as a dedupe key.

#### 5. PubMedQA Split Naming Must Be Explicit

The current docs still say `pqa_labeled` test split. Hugging Face exposes `pqa_labeled` as a single 1,000-row `train` split, while the PubMedQA dataset card notes that 500 of those labeled questions are used as the official test set.

Decision from Claude Code is acceptable: use all 1,000 expert-labeled examples. But document it honestly:

> Evaluation uses all 1,000 expert-labeled `pqa_labeled` rows as an expert-labeled evaluation set, not the official 500-question PubMedQA leaderboard test split.

Also update loader tasks so it does not try `load_dataset(..., "pqa_labeled", split="test")` unless verified.

#### 6. MedMCQA Counts Are Wrong

The docs still say MedMCQA has 194,000 train examples. The Hugging Face dataset card lists:

- train: 182,822
- validation: 4,183
- test: 6,150

The overall dataset is "more than 194k"; the train split is not 194k. This affects the combined training count. The current 405K number should be revised to roughly:

```text
211,269 PubMedQA artificial + 182,822 MedMCQA train = 394,091 examples
```

If the project keeps rounded numbers, label them as approximate.

#### 7. Contamination Check Must Define Fallback Behavior

Claude Code adds PMID overlap checking but does not say what happens if PubMedQA artificial examples do not expose the same field name or type cleanly.

Add this policy:
- Primary dedupe key: `pubid`/PMID if present.
- Secondary dedupe key: normalized question text hash.
- Tertiary check: normalized first context paragraph hash.
- If overlap is found, remove overlapping examples from training and write a contamination report to `metrics/contamination_pubmedqa.json`.

#### 8. Architecture Still Says Dataset Loader Never Raises

The review says loaders should fail loudly, but `architecture.md` still says `Never raises — returns [] and logs to stderr`. That is actively dangerous for this benchmark.

Update architecture:

> Dataset/model loading functions must raise on load, schema, or empty-dataset errors. Only CLI entry points catch exceptions to print readable stderr messages and exit nonzero.

#### 9. Training Validation Split Needs Exact Scope

Claude Code proposes 5% validation split from combined training. Make this explicit:
- create validation after contamination removal
- stratify by task
- seed = 42
- validation used only for training monitoring/early stopping
- final reported metrics never use this validation split

Also decide early stopping patience. Suggested:

```text
eval_steps: 500
save_steps: 500
early_stopping_patience: 3 evals
metric_for_best_model: eval_loss
```

#### 10. Assistant-Only Loss Masking Must Be Tool-Specific

Docs should not only say "assistant-only loss masking." They should say how:
- If using TRL `SFTTrainer`, use a completion/assistant-only data collator compatible with the installed TRL version.
- Add a unit test or dry-run assertion that labels for prompt tokens are `-100` and answer tokens are not.

This is high risk because chat-template masking APIs differ across TRL versions.

#### 11. `versions.lock` Should Include More Than Package Versions

Claude Code proposes `versions.lock`; make its contents explicit:

```yaml
python: "3.11.x"
cuda_module: "<actual CHPC module>"
pytorch: "<version>"
transformers: "<version>"
datasets: "<version>"
peft: "<version>"
trl: "<version>"
bitsandbytes: "<version>"
llama_cpp_python: "<version>"
llama_cpp_commit: "<git sha>"
model_id: "meta-llama/Llama-3.1-8B-Instruct"
model_revision: "<HF revision sha>"
pubmedqa_revision: "<HF dataset revision sha>"
medmcqa_revision: "<HF dataset revision sha>"
prompt_template_version: "v1"
seed: 42
```

#### 12. `CLAUDE.md` Still Has The Old Pitch

`CLAUDE.md` still says:

> The quantization sweet spot for medical LLMs — and why fine-tuned models find it in a different place than base models.

This pre-claims the result. Replace it with the benchmark framing:

> Benchmarking where GGUF quantization preserves or erases medical QA gains from QLoRA fine-tuning.

#### 13. Context-Bridge Ground Truth And Scoring Need More Than Schema

Extending the schema is good, but the scoring logic must change too. Current scoring still gives 1 point for any file overlap and simple substring decisions.

Suggested CB eval scoring:
- files: recall of files_touched, not any-hit
- artifacts: exact/partial path match for artifacts_created and scratch_paths
- decision: semantic substring/keyword overlap, not exact full substring
- next_step: token overlap plus required noun/verb overlap
- blockers/job IDs: exact match when present
- continuity level: exact match

Keep the old 0-4 score only as a compatibility summary if needed.

#### 14. CHPC Setup Still Verifies GPU On The Login Node

`CHPC-setup.md` still says to run CUDA visibility checks directly after creating the venv. On many clusters login nodes do not expose GPUs.

Add:

```bash
salloc --account=yqu-gpu-np --partition=yqu-gpu-np --gpus=1 --time=00:30:00
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use the actual CHPC syntax if it differs.

#### 15. llama-cpp-python CUDA Build Instructions Are Still Missing

`pip install llama-cpp-python` may install CPU-only. Add explicit CUDA build instructions:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir llama-cpp-python
```

Then verify GPU offload in a smoke test, not just package import.

#### 16. HF Offline Mode Is Still Incomplete

Docs still only set `TRANSFORMERS_OFFLINE=1`. Add:

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Also make sure dataset downloads happen before compute-node jobs.

#### 17. Session 1 Has A Prerequisite Contradiction

`session-plan.md` lists "Git repo initialized and pushed to GitHub" as a prerequisite for Session 1, but Session 1 task 1.0 says to initialize the repo. Make it a Session 1 task, not a prerequisite.

#### 18. Session 4 Verification Still Assumes Monotonic Accuracy

`verify/HOW_TO_VERIFY.md` says accuracy should decrease as quantization becomes more aggressive. That is usually true but not guaranteed on small/noisy evals; Q8/Q6/Q4 can tie or occasionally invert by noise.

Replace with:

> Large inversions should be investigated, but small non-monotonic differences can occur. The primary check is that results are complete, finite, and comparable across a single backend.

#### 19. Strong Bar Still Pre-Claims Q4_K_M

`project-spec.md` strong bar still says "Efficiency frontier chart shows Q4_K_M as the sweet spot." Change to:

> Efficiency frontier identifies a deployment sweet spot, whether or not it is Q4_K_M.

#### 20. `DECISIONS.md` Should Not Stay Empty

The project has already made real pre-implementation decisions:
- all primary evals use GGUF/llama.cpp
- primary metric is degradation-from-F16
- PubMedQA uses all 1,000 expert-labeled rows
- Mac fallback uses MLX-LM or is a separate study
- parallel eval writes per-variant raw files then merges

Add these to `docs/DECISIONS.md` before Session 1, or future sessions will keep rediscovering them.

### Required Main-Doc Update Checklist

Before writing code, update the authoritative docs as follows.

#### `CLAUDE.md`

- Replace the one-line pitch with benchmark framing.
- Replace Mac fallback config with MLX-LM wording or mark fallback as separate reduced study.
- Extend CB ground truth schema.
- Update project rules so result records include `run_id`, `eval_backend`, `prompt_template_version`, and `timestamp`.

#### `docs/project-spec.md`

- Soften novelty claim.
- Fix PubMedQA split wording.
- Fix MedMCQA train count and combined training count.
- Add contamination check.
- Add 5% training validation split.
- Replace greedy first-token extraction with direct label scoring plus fallback.
- Add brittleness metrics: `accuracy_abs`, `drop_from_f16`, `brittleness_delta`.
- Separate accuracy scoring from latency measurement.
- Remove Q4_K_M pre-claim from success criteria and interview narrative.

#### `docs/architecture.md`

- Add `src/common/config.py` or explicitly decide no shared config module.
- Change loader behavior from "never raises" to fail-fast.
- Add contamination report output.
- Add assistant-only loss masking and `prepare_model_for_kbit_training()`.
- Add base model GGUF conversion path.
- Add explicit `--outtype f16`.
- Replace mixed backend eval with all-GGUF llama.cpp eval.
- Specify label scoring implementation requirements and spike.
- Update `EvalResult` and `results.json` schema.
- Replace timestamp dedupe with stable run key.
- Add per-variant raw result files plus merge script.
- Add `launch_eval.sh` if parallel eval remains the plan.

#### `docs/session-plan.md`

- Remove Git repo init from Session 1 prerequisites.
- Add `versions.lock` creation.
- Add contamination check task and test.
- Add assistant-mask dry-run assertion.
- Add base GGUF conversion tasks.
- Add logprob scoring spike before full eval.
- Replace HF F16 eval task with GGUF F16 eval.
- Add raw-result merge task for parallel eval.

#### `docs/CHPC-setup.md`

- Add interactive GPU allocation for GPU verification.
- Add `HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.
- Add CUDA build instructions for llama-cpp-python.
- Replace placeholder `cuda/12.x` examples with "verify with module spider and record exact module in versions.lock."
- Fix PubMedQA `pqa_labeled` split command after loader decision is verified.

#### `verify/HOW_TO_VERIFY.md`

- Update CB scoring description for extended schema.
- Update Session 4 verification so it does not require strict monotonic accuracy.
- Add checks for `drop_from_f16`, `brittleness_delta`, `eval_backend`, `run_id`, and `prompt_template_version`.

#### `docs/DECISIONS.md`

- Add pre-implementation decision entries for the major resolved questions listed above.

### Updated Open Questions

The previous five open questions are mostly resolved. These are the new open questions that remain:

1. **Direct label scoring implementation:** will llama-cpp-python candidate scoring be implemented through `logits_all=True` + prompt echo/logprobs, low-level logits access, or another stable API?
2. **Label tokenization policy:** should labels be scored as bare tokens (`A`) or leading-space tokens (` A`), and how will this be validated per model?
3. **Parallel eval persistence:** per-variant raw JSON files plus merge script, or direct locked writes to one `results.json`?
4. **Early stopping patience:** what exact patience/eval_steps/save_steps should be used?
5. **Mac fallback scope:** is MLX-LM fallback still part of v1, or should fallback be documented as a separate stretch study given CHPC A800 access is available?

### Second-Pass Bottom Line

The conceptual review has converged. The remaining risk is not the research idea; it is doc drift. The authoritative docs still instruct an implementation path that contradicts the resolved decisions in this file.

Do not start Session 1 coding until the main docs are updated. Otherwise Claude Code may correctly follow stale instructions and rebuild the broken original design.

---

## Codex Third-Pass Audit After Main Doc Updates

Written: 2026-04-30

Purpose: review the updated MedQuant `.md` files after the main docs were revised from the prior Codex/Claude Code review. This pass checks whether the resolved decisions actually made it into the authoritative docs, and whether any new blindspots were introduced.

### Overall Third-Pass Verdict

**APPROVE TO START AFTER A SMALL DOC CLEANUP PASS.**

The plan is much stronger than the original. The fatal eval-backend problem is mostly fixed, the Mac fallback no longer depends on bitsandbytes/MPS, the PubMedQA split issue is explicit, and the session plan now contains a required logprob-scoring spike before full eval.

Remaining issues are not conceptual blockers, but they matter because stale wording could still make Claude Code implement the wrong thing. The largest remaining risks are:

1. `project-spec.md` still contains stale primary-analysis wording based on raw fine-tuned-minus-base accuracy.
2. `architecture.md` still lacks a true `run_id` in `EvalResult`/dedup keys, making repeat runs ambiguous.
3. The logprob scoring method is still described as more settled than it is; the spike is in `session-plan.md`, but the spec/architecture still read like the API behavior is already known.
4. Context-Bridge dogfood protocol was moved out of `CLAUDE.md` but not made read-first, so it may be skipped.

### What Is Now Fixed Well

These previous concerns are now addressed well enough:

- **Single backend:** `project-spec.md`, `architecture.md`, and `session-plan.md` now specify all primary eval variants as GGUF through llama.cpp/llama-cpp-python.
- **Mac fallback:** `CLAUDE.md` and `project-spec.md` now use MLX-LM rather than bitsandbytes QLoRA on MPS.
- **PubMedQA split:** `project-spec.md`, `docs/dataset.md`, and `CHPC-setup.md` explicitly state that `pqa_labeled` is loaded with `split="train"`, not `split="test"`.
- **MedMCQA reporting:** `docs/dataset.md` correctly says validation is held-out eval, not a hidden test set.
- **Contamination control:** PubMedQA PMID overlap check is planned.
- **Training hygiene:** `prepare_model_for_kbit_training`, assistant-only masking, left truncation, and internal validation split are now documented.
- **Parallel eval:** the plan now uses per-variant raw JSON files plus a merge step, avoiding concurrent writes to one shared `results.json`.
- **DECISIONS.md:** the key pre-implementation decisions are now logged.

### Remaining Issues And Required Updates

#### 1. `project-spec.md` Still Has The Old Primary Analysis

Problem: `project-spec.md` now defines `drop_from_f16` and `brittleness_delta`, but the Experiment Design section still says:

> Primary analysis: for each quantization level, compute the accuracy delta (fine-tuned minus base).

That is the old confounded metric. It contradicts the updated Eval Metrics section and DECISIONS.md.

Required update:

```text
Primary analysis: for each quantization level and dataset, compute drop_from_f16
separately for base and fine-tuned models, then compute brittleness_delta =
finetuned_drop_from_f16 - base_drop_from_f16. Raw fine-tuned-minus-base accuracy
delta is secondary context only.
```

Reason: without this change, the central research question can still drift back toward raw accuracy comparisons rather than quantization sensitivity.

#### 2. Hypothesis Still Pre-Claims Specific Sweet Spots

Problem: `project-spec.md` still predicts:

> The sweet spot is at Q4_K_M for fine-tuned models and Q2_K for base models.

The later interview narrative correctly warns not to pre-claim Q4_K_M. The hypothesis section should be consistent.

Required update: soften this to:

```text
Expected outcome: Q4_K_M may be a practical deployment point for the fine-tuned
model, but the experiment will identify the actual frontier empirically.
```

Reason: a pre-claimed sweet spot weakens credibility if the actual result differs.

#### 3. F16 Base Model Wording Is Still Ambiguous

Problem: `project-spec.md` says:

> The base model F16 is downloaded directly.

That is misleading. The base HF model is downloaded in BF16, but the base F16 eval artifact should be produced by converting the downloaded base HF directory to F16 GGUF, just like the fine-tuned merged model.

Required update:

```text
Base F16 means the downloaded base HF weights converted to F16 GGUF with
convert_hf_to_gguf.py --outtype f16. Fine-tuned F16 means the merged LoRA
weights converted to F16 GGUF with the same command path.
```

Reason: all primary eval artifacts must be GGUF to keep backend/tokenization consistent.

#### 4. `convert_hf_to_gguf.py` Default Rationale Is Incorrect

Problem: `architecture.md` says:

> Without --outtype f16, convert_hf_to_gguf.py defaults to f32.

Current llama.cpp docs list `--outtype f16` as the default for `convert_hf_to_gguf.py`. Keeping the explicit flag is still good, but the stated reason is wrong.

Required update:

```text
Pass --outtype f16 explicitly even though current llama.cpp defaults to f16,
because defaults can change and the result manifest should make the precision
choice auditable.
```

Reason: wrong operational claims tend to confuse future debugging. The explicit flag is correct; the rationale should be version-stable.

Source checked: llama.cpp conversion docs list `--outtype TYPE ... default: f16`.

#### 5. `run_id` Is Still Missing From Results Schema And Dedup Key

Problem: `architecture.md` says the dedup key is:

```text
model_variant + dataset + quantization + hardware
```

This still cannot distinguish:
- two training runs with different LoRA checkpoints
- a full-dataset run vs capped-data run
- a prompt template revision
- a rerun after a llama.cpp commit change

Required update:

```text
Dedup key: run_id + model_variant + dataset + quantization + eval_backend
+ prompt_template_version + hardware
```

Add `run_id` to `EvalResult`, raw JSON files, `results.json`, and `versions.lock`.

Reason: timestamp should not be a dedupe key, but a stable run identifier is necessary for reproducible benchmarking.

#### 6. Config Schema Is Still Stale

Problem: `architecture.md` config schema still describes both configs as sharing BitsAndBytes fields:

```yaml
quantization_bits
bnb_4bit_compute_dtype
bnb_4bit_quant_type
max_samples
```

But `CLAUDE.md` now says Mac config uses:

```yaml
framework: mlx
max_samples_per_dataset
```

Required update:
- Add `framework: trl | mlx`
- Rename/replace `max_samples` with `max_samples_per_dataset`
- Mark BitsAndBytes fields as GPU/TRL-only
- Add reproducibility fields or paths if they are config-driven: `run_id`, `prompt_template_version`, `llama_cpp_dir`, `base_gguf_dir`, `finetuned_gguf_dir`

Reason: config is supposed to drive everything. A stale schema will cause implementation drift immediately in Session 1.

#### 7. Session 1 Directory Tasks Miss Required New Directories

Problem: `session-plan.md` Session 1 creates `metrics/charts/`, but `architecture.md` now expects:

- `src/common/`
- `metrics/raw/`
- `metrics/versions.lock`
- `verify/ground_truth/`

Required update: add `src/common`, `metrics/raw`, and initial `metrics/versions.lock` creation to Session 1. Also add `mlx-lm` to dependencies or clearly mark it Mac-only optional.

Reason: the plan currently asks Session 1 to scaffold the old directory structure, not the revised one.

#### 8. Logprob Scoring Is Still Too Confident In Spec/Architecture

Problem: `session-plan.md` correctly requires a logprob scoring spike, but `project-spec.md`, `docs/dataset.md`, and `architecture.md` still describe direct candidate scoring as if the API behavior is already settled.

Specific issue: llama-cpp-python exposes completion logprobs, but direct multiple-choice label scoring may require `logits_all=True`, prompt echo, low-level logits access, or separate candidate continuation scoring. `logprobs=True` is not precise enough as an implementation spec; OpenAI-style APIs usually expect an integer top-k count, and a boolean may not return enough candidates for A/B/C/D.

Required update:

```text
Accuracy scoring method is direct label scoring through llama-cpp-python, but
the exact API path is decided by verify/check_logprob_scoring.py in Session 4.
The scorer must prove it can score all candidate labels, including leading-space
token variants and multi-token labels if present. Full eval is blocked until
this spike passes.
```

Reason: the spike is the right safeguard, but implementation docs should not over-promise a specific `logprobs=True` mechanism.

Sources checked:
- llama-cpp-python API/discussions indicate logprobs are requested via completion parameters and may require logits availability.
- TRL docs similarly warn template tokenization can differ depending on context; candidate/response token IDs should be tested rather than assumed.

#### 9. Label Tokenization Policy Needs To Move From Session Plan Into Architecture

Problem: `session-plan.md` correctly says to test bare labels vs leading-space labels. `architecture.md` still hardcodes:

```text
PubMedQA: ["yes", "no", "maybe"]
MedMCQA: ["A", "B", "C", "D"]
```

Required update:

```text
Candidate label forms are selected by check_logprob_scoring.py. Test both bare
and leading-space forms; use the form with valid single-token coverage for all
labels, or use multi-token summed logprobs if needed. Record the chosen forms
in metrics/versions.lock.
```

Reason: scoring `A` vs ` A` can change the measured probabilities. This must be fixed before evaluation and recorded for reproducibility.

#### 10. Wilson CI Is Not Enough For The Central Metric

Problem: Wilson intervals on raw accuracy are useful, but the central research finding is `brittleness_delta`, a difference of drops across paired evaluations on the same examples. The current plan does not quantify uncertainty for `drop_from_f16` or `brittleness_delta`.

Required update: add a paired bootstrap over eval examples for:
- `drop_from_f16`
- `brittleness_delta`

Report a 95% bootstrap CI for brittleness_delta per dataset and quantization level.

Reason: raw accuracy CIs do not tell whether the observed brittleness delta is larger than eval noise. The paired setup is an advantage; use it.

#### 11. MedMCQA Filtering Could Change The Eval Count

Problem: docs say MedMCQA eval is n=4,183 and also say filter `choice_type == "single"` before eval. If any validation rows are multi-select, n will be less than 4,183. The docs should not hardcode both without verification.

Required update:

```text
Report MedMCQA validation n after choice_type == "single" filtering. Expected
value is verified in Session 1 and written to versions.lock.
```

Reason: result labels and confidence intervals need the actual evaluated sample count.

#### 12. MedMCQA "No Contamination Risk" Is Too Strong

Problem: `docs/dataset.md` says no MedMCQA deduplication is needed. Even if official splits are intended to be independent, a cheap question-text overlap check is prudent.

Required update: add a Session 1 check for normalized question ID/text overlap between MedMCQA train and validation. Expected overlap should be zero; log it to `metrics/contamination_medmcqa.json` or versions.lock.

Reason: this is low-cost and protects the benchmark from accidental leakage.

#### 13. Training Loss Masking Needs A Version-Specific Test

Problem: docs say to use `DataCollatorForCompletionOnlyLM`, but TRL docs warn completion-only masking only works with `packing=False`, and response template tokenization can fail if tokenized without the same context.

Required update:
- Set `packing=False` explicitly.
- Add a dry-run assertion that prompt token labels are `-100` and answer token labels are not.
- If using Llama-3.1 chat template, derive response template token IDs from the actual rendered chat template instead of assuming a raw string works.

Reason: if assistant masking silently fails, the model trains heavily on long PubMedQA contexts instead of answer tokens.

Source checked: Hugging Face TRL SFTTrainer docs for `DataCollatorForCompletionOnlyLM` completion-only training and response-template tokenization caveat.

#### 14. Latency Definition Is Inconsistent Across Docs

Problem:
- `project-spec.md` says measure TTFT p50/p95 across the eval set.
- `session-plan.md` says latency pass uses 100 prompts.
- Original spec mentioned TPS, but current architecture only stores latency p50/p95 ms.

Required update: choose and document one latency protocol:

```text
Latency is a separate generation pass on 100 fixed prompts per dataset per variant.
Record ttft_p50_ms, ttft_p95_ms, tokens_per_second_p50, tokens_per_second_p95,
context_length_mean, generated_tokens.
```

Or intentionally drop TPS. But do not leave TTFT/TPS mismatch between docs.

Reason: latency numbers are part of the project’s portfolio value; vague measurement makes them hard to defend.

#### 15. Accuracy Pass May Require More Memory Than Generation Pass

Problem: direct scoring may require `logits_all=True`, which increases memory usage. On A800 this is probably fine, but for large prompts and F16 GGUF it should be tested.

Required update: in the Session 4 spike, require testing the largest F16 GGUF and a long PubMedQA prompt, not only the smallest/safest GGUF.

Reason: a scorer that works on Q4 with short prompts can still OOM or fail on F16 with long abstracts.

#### 16. Context-Bridge Dogfood Can Still Be Missed

Problem: `verify/MEDQUANT-SESSION-CHECKLIST.md` is good, but `CLAUDE.md` does not list it in the Document Index, and `session-plan.md` no longer includes CB instrumentation tasks despite Session 1 still being titled "CB instrumentation."

Required update:
- Add `verify/MEDQUANT-SESSION-CHECKLIST.md` to the `CLAUDE.md` Document Index.
- Add a Session 1 checklist item to verify `.mcp.json`, `/bridge`, SessionEnd hook, and `verify/eval_cb_packet.py`.
- Add a per-session wrap-up checkbox: update ground truth and dogfood log.

Reason: if Gate E dogfood is still a goal, it needs to be part of the read-first workflow, not a side document that can be forgotten.

#### 17. `CLAUDE.md` Project Rule 5 Is Stale

Problem: `CLAUDE.md` still says every result entry must include:

```text
model_variant, hardware, timestamp, n_samples
```

Required update:

```text
Every result entry must include run_id, model_variant, quantization, dataset,
n_samples, hardware, eval_backend, prompt_template_version, timestamp, and
source artifact path/hash.
```

Reason: provenance now matters more than in the original design.

#### 18. `versions.lock` Is Listed Under `metrics/`, But `metrics/` Is Not Created Yet

Problem: the repo currently has no `metrics/` directory. That is fine pre-Session 1, but Session 1 must create it before writing `versions.lock`.

Required update: add `.gitkeep` files or explicit Session 1 tasks for:

- `metrics/raw/`
- `metrics/charts/`
- `metrics/versions.lock`

Reason: otherwise the reproducibility lock step can fail just because the path does not exist.

#### 19. Full Eval "Run Exactly Once" Needs Failure Policy

Problem: `docs/dataset.md` says eval runs exactly once and no rerunning after seeing results. Good principle, but operationally you need to allow reruns for failed jobs, corrupted raw files, or scorer failure before results are inspected.

Required update:

```text
Allowed reruns: infrastructure failure, invalid_output_rate > 0 due to scorer/API
failure, missing raw file, corrupted GGUF, or job preemption. Reruns must be logged
in metrics/eval_reruns.md with reason and affected variant. Not allowed: rerunning
because a result looked surprising.
```

Reason: this prevents both accidental p-hacking and paralysis when CHPC jobs fail.

#### 20. CHPC llama-cpp-python GPU Verification Is Too Weak

Problem: `CHPC-setup.md` verifies only that `from llama_cpp import Llama` imports. That does not prove CUDA offload is active.

Required update: add a real tiny GGUF load/inference smoke test with `n_gpu_layers=-1` and logs inspected for CUDA/cuBLAS/GGML_CUDA. If no tiny GGUF exists yet, run this in Session 4 with the first converted model.

Reason: a CPU-only llama-cpp-python install can silently make eval far slower and distort latency.

### Recommended Priority Order

#### P0 — Fix Before Any Coding

1. Update `project-spec.md` primary analysis to brittleness_delta, not raw fine-tuned-minus-base delta.
2. Fix `project-spec.md` F16 base wording.
3. Update `architecture.md` config schema for `framework`, `max_samples_per_dataset`, and GPU-only BitsAndBytes fields.
4. Add `run_id` to `EvalResult`, `results.json`, raw result files, dedup key, and `versions.lock`.
5. Add `verify/MEDQUANT-SESSION-CHECKLIST.md` to the `CLAUDE.md` Document Index if CB dogfood is still required.

#### P1 — Fix Before Session 2 Training

6. Add `packing=False` and masking assertions for `DataCollatorForCompletionOnlyLM`.
7. Add MedMCQA train/validation text-overlap check.
8. Make validation split stratified by task and record exact counts.
9. Create/commit `metrics/versions.lock` only after real versions and dataset sizes are known; include `run_id`.

#### P2 — Fix Before Session 4 Eval

10. Move label-tokenization policy from Session 4 checklist into `architecture.md`.
11. Require scoring spike on F16 plus a long PubMedQA prompt, not only a small GGUF.
12. Add paired bootstrap CIs for `drop_from_f16` and `brittleness_delta`.
13. Resolve latency protocol: 100 prompts vs full eval set, TTFT only vs TTFT+TPS.
14. Add eval rerun policy.
15. Add real llama-cpp-python GPU offload verification.

### Updated Bottom Line

The plan is now fundamentally sound. The remaining blindspots are mostly reproducibility and implementation-contract issues, not project-killing research flaws.

The two things I would not leave ambiguous are `run_id` and logprob scoring. Without `run_id`, the results become hard to compare across reruns. Without a stricter scorer contract, Session 4 can still burn time discovering that `logprobs=True` does not directly produce the candidate probabilities the plan assumes.
