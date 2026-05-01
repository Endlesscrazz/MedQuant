# MedQuant — Verification Script Inventory
# Written: 2026-04-27
# Purpose: one entry per session describing what to run, what to look for,
#          and what "correct" looks like.
# ──────────────────────────────────────────────────────────────────────────

## HOW TO USE

Run the verification for a session BEFORE marking it complete.
Real-data verification is not optional — passing tests alone is insufficient.

---

## Session 1 verification — data pipeline smoke test

```bash
pytest tests/test_data.py -v
```

What to look for:
- All tests pass (loader + formatter + contamination check tests)
- No import errors
- Dataset schema assertions pass

Real-data check:
```bash
python -c "
from src.data.loader import load_pubmedqa, load_medmcqa
pub_train = load_pubmedqa('train')
pub_eval  = load_pubmedqa('eval')
med_train = load_medmcqa('train')
med_eval  = load_medmcqa('validation')
print(f'PubMedQA train: {len(pub_train)}')
print(f'PubMedQA eval:  {len(pub_eval)}')
print(f'MedMCQA train:  {len(med_train)}')
print(f'MedMCQA eval:   {len(med_eval)}')
print('Sample answer:', pub_eval[0]['answer'])
"
```

Expected: PubMedQA eval = 1000, answer is yes/no/maybe. MedMCQA eval = 4183.
Contamination check logs to stderr: "removed N samples (pubid overlap)".

Check versions.lock was written and committed:
```bash
cat metrics/versions.lock
```

---

## Session 2 verification — training dry-run

```bash
python src/train/train.py --config config/gpu_config.yaml --dry-run
```

What to look for:
- No errors loading model + tokenizer
- Batch shape printed (matches max_seq_length)
- Loss value printed (finite, not NaN)
- Script exits cleanly without saving anything
- On CHPC: same output with real Llama-3.1-8B weights loaded

After smoke job (1K samples, 1 epoch) completes:
```bash
cat outputs/logs/<run_name>/training_log.json | python -m json.tool | head -40
```
Look for: decreasing train_loss, reasonable val_loss, no NaN values.
Record wall time for 1K samples to decide if full-dataset cap is needed.

---

## Session 3 verification — GGUF files

```bash
# Check all 10 files exist
ls -lh $SCRATCH/models/base-gguf/
ls -lh $SCRATCH/models/finetuned-gguf/

# Quick inference test on each GGUF
for f in $SCRATCH/models/finetuned-gguf/*.gguf; do
    echo "=== $f ==="
    ./llama.cpp/build/bin/llama-cli -m "$f" \
        -p "Question: Is ibuprofen an NSAID? Answer yes, no, or maybe.\n" \
        -n 3 --temp 0 2>/dev/null | tail -2
done
```

What to look for:
- 5 base GGUF + 5 finetuned GGUF = 10 files total
- Each produces non-empty text output (correctness doesn't matter — load check only)
- No segfaults or load errors
- Approximate file sizes: F16 ~16GB, Q8 ~8.5GB, Q6 ~6.1GB, Q4 ~4.7GB, Q2 ~2.7GB

---

## Session 4 verification — results table

```bash
python -c "
import json
results = json.load(open('metrics/results.json'))
print(f'Total records: {len(results)} (expect 20)')
for r in results:
    print(f\"{r['model_variant']:30s} {r['dataset']:10s} acc={r['accuracy_abs']:.3f} drop={r['drop_from_f16']:.3f}\")
"
```

What to look for:
- 20 records total (10 variants × 2 datasets)
- finetuned_f16 accuracy > base_f16 on both datasets (if not, training failed)
- Large accuracy inversions (e.g., Q4 beats Q8 by > 3%) should be investigated —
  they may indicate a corrupted GGUF or scorer failure, not a real quantization effect.
  Small non-monotonic differences at adjacent levels (e.g., Q8 ≈ Q6 ± 1%) are expected
  noise and are not verification failures.
- No NaN or 0.0 accuracy values
- invalid_output_rate = 0.0 for all variants
- drop_from_f16 and brittleness_delta populated (not -1.0 sentinel)

---

## Session 5 verification — charts and serve

Charts:
```bash
ls -lh metrics/charts/
open metrics/charts/quality_cliff_pubmedqa.png
```
What to look for: 4 PNG files, two visible lines per chart, annotated deltas, labeled axes.

Serve:
```bash
GGUF_MODEL_PATH=$SCRATCH/models/finetuned-gguf/q4_k_m.gguf \
    uvicorn src.serve.app:app --reload

# In another terminal:
curl -X POST http://localhost:8000/answer \
    -H "Content-Type: application/json" \
    -d '{"question": "Is aspirin an anticoagulant?", "context": null, "task": "pubmedqa"}'
```
What to look for: JSON response with "answer" field = "yes", "no", or "maybe".
Response time < 5 seconds on local Mac.
