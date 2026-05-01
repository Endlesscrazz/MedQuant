# MedQuant — Dataset Documentation
# Written: 2026-04-28
# Purpose: authoritative reference for all dataset decisions — what datasets
#          we use, why, how splits are defined, contamination controls,
#          eval protocol, and reporting conventions.
#          Changes to any decision here require a DECISIONS.md entry.
# ──────────────────────────────────────────────────────────────────────────

## WHY THESE TWO DATASETS

Both datasets are freely available on HuggingFace with no credentialing or
institutional data use agreements required. Both have published baselines from
GPT-4 and Llama-family models, enabling direct comparison against literature.
Both are text-only question answering, keeping the fine-tuning task clean and
evaluation deterministic.

They cover complementary angles of medical language understanding:
- PubMedQA: biomedical research question answering (3-class, evidence-based)
- MedMCQA: clinical medicine multiple choice (4-class, USMLE-style)

Using both mitigates the risk of a model specializing to a single answer
format. The training sizes are naturally close (~211K vs ~183K), avoiding
the need for resampling or class-balancing logic.

---

## DATASET 1 — PubMedQA

### What it is

PubMedQA is a biomedical research question answering dataset. Given a research
question and the abstract of a PubMed article, the task is to answer yes, no,
or maybe. The question is answerable from the abstract alone; the abstract
serves as the context passage. This tests whether a model can read biomedical
literature and draw a supported conclusion.

Paper: Jin et al., "PubMedQA: A Dataset for Biomedical Research Question
Answering," EMNLP 2019.
HuggingFace ID: qiaojin/PubMedQA

### Three subsets and their actual HuggingFace split names

PubMedQA has three subsets accessed via the `name` parameter. The split names
inside each subset are non-obvious and must not be assumed.

| Subset name | How to load | Size | Labels | Split name inside |
|---|---|---|---|---|
| pqa_labeled | load_dataset("qiaojin/PubMedQA", "pqa_labeled") | 1,000 | Expert-annotated by biomedical researchers | **"train"** |
| pqa_artificial | load_dataset("qiaojin/PubMedQA", "pqa_artificial") | 211,269 | Auto-generated from structured PubMed abstracts | "train" |
| pqa_unlabeled | load_dataset("qiaojin/PubMedQA", "pqa_unlabeled") | 61,249 | Expert-collected, no answer labels | "train" |

CRITICAL: pqa_labeled does NOT have a "test" split. The 1,000 expert-labeled
evaluation samples are accessed via split="train" within the pqa_labeled subset.
Calling split="test" will raise a KeyError. Verify this in Session 1 before
writing any loader code.

Correct loading pattern:
```python
# Training data:
train_ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

# Evaluation data (1K expert-labeled):
eval_ds  = load_dataset("qiaojin/PubMedQA", "pqa_labeled",    split="train")
```

### Key fields

| Field | Type | Description | Used in |
|---|---|---|---|
| pubid | int | PubMed article identifier | Contamination check |
| question | str | The research question | Prompt construction |
| context | dict | Contains `contexts` (list of sentences from abstract) | Prompt construction |
| final_decision | str | Gold label: "yes", "no", or "maybe" | Eval target |
| long_answer | str | Full-text explanation | NOT used |

For prompt construction: join `context['contexts']` with spaces as the context
passage. Do not use `long_answer` in training or eval prompts — it contains the
answer reasoning and would leak the label.

### Why pqa_artificial for training

pqa_artificial (211K) was auto-generated from PubMed abstracts that contain
structured conclusion sections explicitly stating a yes/no/maybe outcome. The
auto-generation quality is high because the source material is structured
scientific writing, not crowd-sourced annotation. It provides a large and
diverse training signal without requiring expert labeling at scale.

pqa_unlabeled is NOT used for training — it has no answer labels and adding it
as unsupervised signal would complicate the SFT training objective.

### Eval split: all 1,000 expert-labeled samples

Decision: use all 1,000 expert-labeled samples as a single eval set.
No 500/500 dev/test split.

Reasoning:
1. We do no prompt tuning, threshold calibration, or hyperparameter selection
   against the eval set. Logprob scoring (see Eval Protocol) selects the argmax
   directly — there is no tunable decision boundary.
2. Splitting 1,000 to 500 halves the sample size and widens confidence intervals
   (from ±~3% to ±~4.4% at 95% confidence around 70% accuracy) for no gain.
3. The eval set is run exactly once per model variant at the end of the project.
   No iterative peeking means no need to hold out a final test set.

Reporting label: "PubMedQA pqa_labeled eval (n=1,000)"

### Published baselines for comparison

| Model | Accuracy | Source |
|---|---|---|
| GPT-4 | ~77–78% | Jin et al. + benchmarks |
| Llama-3.1-8B zero-shot | ~60–65% | Estimated from PMC 2025 studies |
| Llama-3.1-8B SFT (medical) | ~68–73% | PMC 2025 comparative study |

---

## DATASET 2 — MedMCQA

### What it is

MedMCQA is a large-scale medical multiple-choice QA dataset derived from
AIIMS and NEET PG (Indian medical board) entrance examinations. Questions
cover clinical medicine, pharmacology, anatomy, biochemistry, and allied
medical subjects. Format: 4-option single-answer MCQ. Difficulty approximates
USMLE Step 1/2.

Paper: Pal et al., "MedMCQA: A Large-Scale Multi-Subject Multi-Choice Dataset
for Medical domain Question Answering," PMLR 2022.
HuggingFace ID: openlifescienceai/medmcqa

### Split structure

| Split | Loaded size | Labels | Used for |
|---|---|---|---|
| train | ~182,822 | Yes | Fine-tuning |
| validation | 4,183 | Yes | Held-out eval |
| test | 6,150 | No (withheld) | NOT used |

Note: HuggingFace may report 194K for train; the actual loaded size after
deduplication by the dataset authors is ~182,822. Verify with len() in Session 1.

### Why the validation split is our eval set

The test split labels are withheld by the dataset authors and not publicly
available. Using the validation split (4,183 samples with known labels) as
a held-out eval set is the standard approach in the medical NLP community —
this is how most published MedMCQA results are reported.

Reporting label: "MedMCQA validation-as-held-out-eval (n=4,183)"
Do not call this the "test set" anywhere in results, charts, or README.

### Key fields

| Field | Type | Description | Used in |
|---|---|---|---|
| id | str | Unique question ID | Deduplication check |
| question | str | The question text | Prompt |
| opa, opb, opc, opd | str | Four answer options | Prompt |
| cop | int | Correct option index (0=A, 1=B, 2=C, 3=D) | Eval target |
| choice_type | str | "single" or "multi" | Filtering |
| subject_name | str | Medical subject area | Per-subject breakdown |

Filter to `choice_type == "single"` before both training and eval.
Multi-select questions require a different output format and cannot be
evaluated reliably with single-label logprob scoring.

### Published baselines for comparison

| Model | Accuracy | Source |
|---|---|---|
| GPT-4 | ~87% | Pal et al. + benchmarks |
| Llama-3.1-8B zero-shot | ~55–62% | Related work estimates |

---

## TRAINING DATASET CONSTRUCTION

### Combination strategy: natural mix

Steps in order:
1. Load pqa_artificial (split="train") from HuggingFace or local cache
2. Run PubMedQA contamination check — remove any sample whose pubid appears
   in pqa_labeled eval set (see Contamination section below)
3. Load MedMCQA train split, filter to choice_type == "single"
4. Format both datasets into instruction-tuning pairs using the model's
   chat template (see architecture.md formatter spec)
5. Concatenate into one list
6. Shuffle with seed=42
7. Apply max_samples_per_dataset cap if needed (see below)

Natural ratio after filtering: ~211K PubMedQA : ~182K MedMCQA ≈ 1.16:1.
No resampling is applied — the natural ratio is close enough to avoid
systematic task imbalance. Report per-dataset eval results separately to
detect if the training mix hurts one task.

### Max samples cap (for time-budget control)

Default: max_samples_per_dataset: null (use full dataset).

If a 1K-sample timed smoke job on A800 projects total training time
exceeding 5 hours for 2 epochs, apply the cap in gpu_config.yaml:
```yaml
max_samples_per_dataset: 150000   # 150K per dataset → 300K total
```

For Mac MLX fallback:
```yaml
max_samples_per_dataset: 50000    # 50K per dataset → 100K total
```

Subsampling is applied independently per dataset before combining, with seed=42,
to preserve the approximate 1:1 ratio. Record the final sample count in
metrics/versions.lock before training starts.

### Validation split for early stopping

Reserve 5% of the combined training set as an internal validation set for
early stopping signal. This split is created AFTER contamination removal
and BEFORE training, using seed=42.

```
combined_data (post-contamination, post-cap)
    ├── 95% → training set   (used for gradient updates)
    └──  5% → val_early_stop (used only for early stopping loss monitoring)
```

This internal validation set is NOT the PubMedQA or MedMCQA eval sets.
It may contain pqa_artificial samples. It is never used for final accuracy
reporting — only for detecting overfitting during training.

### Loss masking: train on answer tokens only

Apply assistant-token loss masking during SFT. The model should compute loss
only on the answer portion of each training example, not the prompt/context.

If using TRL SFTTrainer: use DataCollatorForCompletionOnlyLM, specifying the
response template that marks the start of the assistant turn. This ensures
the prompt tokens (which can be hundreds of tokens for PubMedQA with its
abstract context) do not dominate the loss.

Truncation: if a formatted example exceeds max_seq_length, truncate from the
LEFT (remove the beginning of the context). This preserves the question and
answer at the end of the sequence, which are the tokens the model must learn.

---

## CONTAMINATION AVOIDANCE

### PubMedQA: PMID-based deduplication

Risk: pqa_artificial and pqa_labeled are both derived from PubMed articles.
A training sample with the same PubMed article ID (pubid) as an eval sample
means the model has seen that article's content during training, which could
inflate eval accuracy.

Control: before combining training data, extract all pubid values from
pqa_labeled (the 1K eval set) and remove matching samples from pqa_artificial.

```python
def check_pubmedqa_contamination(
    train_items: list[dict],
    eval_items: list[dict],
) -> tuple[list[dict], int]:
    eval_pubids = {str(item["pubid"]) for item in eval_items}
    cleaned = [x for x in train_items if str(x["pubid"]) not in eval_pubids]
    removed = len(train_items) - len(cleaned)
    return cleaned, removed
```

Implementation: src/data/contamination_check.py, run during Session 1.
Expected overlap: 0 (dataset authors designed these to be non-overlapping),
but the check must be run and the result logged regardless.

Log to stderr: "PubMedQA contamination check: removed N samples (pubid overlap)"

### MedMCQA: cheap overlap check

MedMCQA train and validation splits are drawn from different exam question
pools. The dataset authors confirm no overlap in the original paper. However,
as a low-cost safeguard, run a normalized question text hash check between
train and validation in Session 1:

```python
import hashlib
def norm_hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().split()).hexdigest()
```

Compare hash sets. Expected overlap: 0. If any are found, remove from train
and log to metrics/contamination_medmcqa.json. Result logged to versions.lock
as medmcqa_train_val_overlap: 0 (or N if found).

### Eval integrity protocol — what we never do

The eval sets (pqa_labeled 1K + MedMCQA val 4,183) are read-only inputs to
the eval harness. They are never used to:
- Select or tune training hyperparameters (use the 5% internal val split)
- Tune or adjust the prompt template (template is fixed in Session 1)
- Filter training data beyond the PMID contamination check
- Inspect individual examples during model development

Eval is run once per model variant in Session 4 and the results are final.
No re-running after inspecting results.

---

## EVAL PROTOCOL

### Inference method: logprob scoring via llama.cpp

All 10 model variants (base + fine-tuned × 5 quantization levels) are
evaluated through a single inference backend: llama.cpp via llama-cpp-python.
All model variants — including the F16 base and fine-tuned references — are
loaded as GGUF files. No HuggingFace Transformers backend is used for primary
accuracy evaluation.

Why logprob scoring instead of greedy text generation:
- Eliminates first-token fragility (whitespace, BOS/EOS artifacts, explanations
  before the answer, especially common at aggressive quantization levels like Q2)
- Consistent across all 10 variants through the same code path
- Deterministic: no sampling randomness
- Works reliably even when Q2 model generation is unstable

Method:
1. Build the full formatted prompt for each eval example
2. Tokenize each candidate answer label separately:
   - PubMedQA: ["yes", "no", "maybe"]
   - MedMCQA: ["A", "B", "C", "D"]
3. Compute the log probability of each candidate token at the answer position,
   conditioned on the full prompt, via llama_cpp logprobs parameter
4. Select argmax — the candidate with highest log probability is the prediction

Fallback: if a GGUF file fails to return logprobs (load error or API failure),
record invalid_output_rate=1.0 for that variant and exclude it from brittleness
analysis with an explicit note in results.json.

### Metrics: what we measure and why

Five metrics per (model_variant × dataset) record:

**accuracy_abs** — raw accuracy on the full eval set.
Answers: how accurate is this model at this quantization level?
Context metric — shows absolute quality, not the research finding.

**drop_from_f16** — accuracy_abs[F16] minus accuracy_abs[Qx] for the same
model family. Computed independently for base and fine-tuned.
Answers: how much quality does this model lose when compressed to level Qx?
Primary metric for each model family.

**brittleness_delta** — finetuned_drop_from_f16 minus base_drop_from_f16
at the same quantization level.
Answers the central research question: is the fine-tuned model more sensitive
to compression than the base model?
  Positive: fine-tuned model degrades more (more brittle)
  Zero: same sensitivity
  Negative: fine-tuned model is more robust under quantization

**confidence_interval_95** — 95% Wilson confidence interval on accuracy_abs,
reported as [lower, upper]. With n=1,000: ±~3% at 70% accuracy. With n=4,183:
±~1.5% at 70% accuracy.

**invalid_output_rate** — proportion of samples where logprob scoring failed.
Expected 0.0 for all variants under normal operation. Non-zero signals a
corrupted GGUF or llama.cpp API change — those results are excluded from
brittleness analysis.

### Per-dataset reporting

All metrics are reported separately for PubMedQA and MedMCQA. Results are
never averaged across datasets. The two tasks have different baselines
(33% vs 25% random chance), different answer formats, and different
distributions. A mixed accuracy number is not interpretable.

### Eval runs exactly once — with explicit rerun policy

Each model variant is evaluated once. No re-running after seeing results to
avoid implicit data-peeking. Exceptions that permit a rerun:

- Infrastructure failure: SLURM job preempted, node crash, OOM
- Scorer failure: invalid_output_rate > 0 due to logprob API or GGUF issue
- Missing raw file: metrics/raw/{variant}.json absent after job completion
- Corrupted GGUF: load or inference failure confirmed on the GGUF file itself

NOT permitted: rerunning because a result looked surprising or lower than expected.

All reruns must be logged in metrics/eval_reruns.md:
  - date, affected variant, reason, action taken

The run timestamp is recorded in results.json as an audit field only.

---

## REPRODUCIBILITY RECORD

Before training begins (end of Session 1), record the following in
metrics/versions.lock and commit to git:

```
# MedQuant reproducibility lock — generated end of Session 1
# DO NOT MODIFY after training starts

# Environment
python_version: "3.11.x"                  # fill in: python --version
cuda_module: "<actual CHPC module>"        # fill in: module spider cuda, then record exact string

# Model
model_id: "meta-llama/Llama-3.1-8B-Instruct"
model_revision: "<HF revision sha>"       # fill in: huggingface-cli download --revision

# Datasets
pubmedqa_dataset_revision: <git commit hash from HuggingFace dataset repo>
medmcqa_dataset_revision: <git commit hash from HuggingFace dataset repo>
pubmedqa_pqa_artificial_size_before_contamination: <N>
pubmedqa_pqa_artificial_size_after_contamination: <N>
pubmedqa_contamination_removed: <N>
medmcqa_train_size_after_filter: <N>
combined_train_size: <N>
val_early_stop_size: <N>
random_seed: 42

# Packages
torch_version: <version>
transformers_version: <version>
peft_version: <version>
trl_version: <version>
bitsandbytes_version: <version>
llama_cpp_python_version: <version>
llama_cpp_commit: <hash>                  # fill in: git -C /path/to/llama.cpp rev-parse HEAD

# Eval
prompt_template_version: v1
```

This file is committed once and never modified after training begins.
Any re-run that changes these values requires a new entry in DECISIONS.md.

---

## REPORTING CONVENTIONS SUMMARY

| Dataset | Eval split | Correct label in reports |
|---|---|---|
| PubMedQA | pqa_labeled (all 1K) | "PubMedQA pqa_labeled eval (n=1,000)" |
| MedMCQA | validation split | "MedMCQA validation-as-held-out-eval (n=4,183)" |

Never use:
- "test set" for MedMCQA (the real test labels are withheld)
- "held-out test" for PubMedQA (we use all 1K for eval)
- Any averaged cross-dataset accuracy
- Numbers without hardware label (A800 or Mac M1 Air)
