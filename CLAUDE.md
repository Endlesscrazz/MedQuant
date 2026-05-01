# MedQuant — Claude Code Context
# /Users/shreyas/Desktop/UoU/Claude-workspace/projects/MedQuant/CLAUDE.md
# ──────────────────────────────────────────────────────────────────────────

## WHAT THIS PROJECT IS

MedQuant is a research pipeline that answers one focused question:
**Does QLoRA fine-tuning on medical domain data make a general LLM more
brittle to GGUF quantization compared to the base model?**

Concretely: fine-tune Llama-3.1-8B-Instruct on PubMedQA + MedMCQA with
QLoRA, then systematically measure how GGUF quantization at five levels
(Q2/Q4/Q6/Q8/F16) affects accuracy, memory, and latency for both the base
and fine-tuned model. The output is a "quality cliff chart" showing where
fine-tuned and base models diverge under compression.

One-line pitch: "Benchmarking where GGUF quantization preserves or erases
medical QA gains from QLoRA fine-tuning."

---

## ⚠️ IMPLEMENTATION BOUNDARY

All code, configs, and docs live in:
  /Users/shreyas/Desktop/UoU/Claude-workspace/projects/MedQuant/

Model weights and GGUF files live on CHPC scratch only:
  /scratch/general/vast/u1527145/medquant/

Never commit model weights, checkpoints, or GGUF files to git.
They are too large and live exclusively on scratch.

---

## CURRENT STATE (update this block after every session)

Sessions complete: 0
Tests passing: N/A (project not started)
Training status: Not started
Baseline accuracy (base Llama-3.1-8B): Not measured
Fine-tuned accuracy: Not measured
Active config profile: GPU (config/gpu_config.yaml)

Next session: Session 1 — Repo setup, data pipeline, CB instrumentation

---

## HARDWARE PROFILES

Two profiles committed to the repo. One flag swap changes the entire pipeline.
Never hardcode model names, batch sizes, or paths — always read from config.

### GPU profile — CHPC A800 80GB (primary)
File: config/gpu_config.yaml
```
model_id: meta-llama/Llama-3.1-8B-Instruct
model_nickname: llama-3.1-8b
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, v_proj, k_proj, o_proj]
batch_size: 32
gradient_accumulation_steps: 1
max_seq_length: 2048
learning_rate: 2.0e-4
num_train_epochs: 2
lr_scheduler: cosine
warmup_ratio: 0.03
quantization_bits: 4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
```

### Mac fallback — M1 Air (if GPU access lost)
File: config/mac_config.yaml
Training framework: mlx-lm (NOT bitsandbytes — unreliable on Apple MPS).
```
model_id: Qwen/Qwen2.5-1.5B-Instruct
model_nickname: qwen-2.5-1.5b
framework: mlx                  # signals train.py to use mlx_lm instead of TRL
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: [q_proj, v_proj]
batch_size: 4
gradient_accumulation_steps: 8
max_seq_length: 1024
learning_rate: 2.0e-4
num_train_epochs: 2
lr_scheduler: cosine
warmup_ratio: 0.05
max_samples_per_dataset: 50000  # 50K × 2 = 100K total (time budget)
```

If GPU access is lost mid-project: switch active config to mac_config.yaml.
Install mlx-lm: pip install mlx-lm. The train.py script reads the `framework`
field from config to branch between TRL (GPU) and mlx_lm (Mac).
Results are smaller-scale but the brittleness_delta question is still answerable.

---

## WORKFLOW: LOCAL DEV + REMOTE TRAINING

Claude Code sessions run entirely on the local Mac.
CHPC is accessed via SSH only to submit jobs and retrieve outputs.

```
Local Mac (Claude Code — all planning, coding, analysis)
    │
    ├── git push → GitHub
    │       │
    │       └── CHPC login node (SSH)
    │               ├── git pull
    │               ├── sbatch slurm/train.sbatch
    │               └── outputs → /scratch/.../medquant/
    │                       │
    │                       └── git push logs + metrics back
    │
    └── git pull → analyze results locally
```

What lives locally (git-tracked): code, configs, tests, logs, metrics JSON.
What lives on scratch only (never git): model weights, GGUF files, checkpoints.

SLURM partition: yqu-gpu-np
CHPC scratch base: /scratch/general/vast/u1527145/medquant/
CHPC home (code only): /uufs/chpc.utah.edu/common/home/u1527145/projects/medquant/

---

## BUILD METHODOLOGY — REQUIRED EVERY SESSION

Every session follows this loop in order. Do not skip steps.

1. Build — implement per architecture.md + session plan
2. Test — pytest tests/ must be 100% green before moving on
3. Verify on real data — run the verify script for the session
4. Update docs — DECISIONS.md (if schema/interface changed), CLAUDE.md CURRENT STATE
5. Update session-plan.md — check off completed tasks, mark session COMPLETE

Never declare a session done with unchecked boxes or failing tests.

---

## DECISION BRIEF — REQUIRED BEFORE EVERY NON-TRIVIAL IMPLEMENTATION

```
DECISION BRIEF: [component name]
──────────────────────────────────────────────────────────
Approach:    [what you're about to build]
Alternative: [what else was considered]
Trade-off:   [what we're accepting]
Interview:   "[how Shreyas explains this in 2-3 sentences]"
→ Proceed?
```

Auto-triggers (always output a brief first):
- New module or script introduced
- Config schema changes
- Dataset preprocessing decisions
- Training hyperparameter choices
- Eval metric design
- Any choice that affects results reproducibility

No brief needed for: bug fixes under 20 lines, adding tests for existing code,
renaming/reformatting, minor config tweaks.

---

## END OF SESSION SUMMARY FORMAT

After completing any feature or full session:

```
✅ DONE: [session N / feature name]

WHAT WAS BUILT: [2–3 sentences]
KEY DECISION:   [the most interesting technical choice]
TRADE-OFF:      [what was given up]

INTERVIEW SUMMARY:
"[3–5 sentences Shreyas can say verbatim in a screen]"

DECISIONS.md: updated ✓ / not needed ✓
Next: [specific first task of next session]
```

---

## MODULE ISOLATION RULES

Inter-module communication happens via file paths (artifacts, JSON),
not direct Python imports between pipeline stages.

- src/data/: no imports from train, eval, convert, serve
- src/train/: imports data utilities only. No imports from eval, convert, serve.
- src/eval/: imports data utilities only. No imports from train, convert, serve.
- src/convert/: no imports from eval, serve, data
- src/serve/: no imports from train, convert, eval, data
- verify/: may import any module — explicitly exempt
- config/: passive YAML files only — no imports

---

## PROJECT RULES

1. config.yaml drives everything. Model name, batch size, paths — all from config.
   No hardcoded values anywhere in training/eval/convert scripts.
2. Never commit model weights, checkpoints, or GGUF files.
   Add to .gitignore at project init and never remove them.
3. pytest for all data processing, formatting, and eval logic.
   Training scripts tested via --dry-run flag (validates pipeline, no actual training).
4. Verbose/debug output to stderr only. Eval results to structured JSON (stdout or file).
5. Every result entry must include: run_id, model_variant, quantization, dataset,
   n_samples, hardware, eval_backend, prompt_template_version, timestamp.
   No orphaned numbers without provenance.
6. SLURM job scripts must set --mail-type=END,FAIL with Shreyas's email.
7. All CHPC scratch paths must be parameterized in config — never hardcoded.
8. DECISIONS.md updated only when a function signature or schema changes.
   Do not log minor implementation choices.

---

## DOCUMENT INDEX

Read-first every session:
| File | Purpose |
|---|---|
| CLAUDE.md (this file) | Session context, rules, hardware profiles, CB integration |
| docs/session-plan.md | Session-by-session task checklists — authoritative tracker |

Reference when building:
| File | Purpose |
|---|---|
| docs/project-spec.md | Problem statement, research question, datasets, success criteria |
| docs/architecture.md | Module specs, data flow, eval harness design, results schema |
| docs/DECISIONS.md | Append-only design decision log. Newest at top. |
| docs/CHPC-setup.md | Cluster env setup, HuggingFace download, SLURM templates |

Operational:
| File | Purpose |
|---|---|
| verify/HOW_TO_VERIFY.md | Per-session verification script inventory |
| verify/MEDQUANT-SESSION-CHECKLIST.md | CB dogfood ground truth per session — update at every session end |
| metrics/results.json | Full eval results table (all 10 model variants) |

## SESSION CONTINUITY (context-bridge)
Type `/bridge` at session start to resume from prior context.
Run `context-bridge simulate` after a session to preview what was captured.
