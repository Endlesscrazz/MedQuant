SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)

# ── Local development ──────────────────────────────────────────────────────────

.PHONY: test
test:
	source medq_env/bin/activate && pytest tests/ -v

.PHONY: test-cov
test-cov:
	source medq_env/bin/activate && pytest tests/ -v --cov=src --cov-report=term-missing

.PHONY: train-dry
train-dry:
	source medq_env/bin/activate && python src/train/train.py --config config/mac_config.yaml --dry-run

# ── CHPC sync ─────────────────────────────────────────────────────────────────

.PHONY: sync-up
sync-up:
	bash scripts/sync_to_chpc.sh

.PHONY: sync-results
sync-results:
	bash scripts/sync_from_chpc.sh

# ── CHPC job submission (run from login node OR via SSH) ───────────────────────

.PHONY: submit-train
submit-train:
	ssh notchpeak "cd /uufs/chpc.utah.edu/common/home/u1527145/projects/medquant && sbatch slurm/train.sbatch"

.PHONY: submit-convert
submit-convert:
	ssh notchpeak "cd /uufs/chpc.utah.edu/common/home/u1527145/projects/medquant && sbatch slurm/convert.sbatch"

.PHONY: submit-eval
submit-eval:
	ssh notchpeak "cd /uufs/chpc.utah.edu/common/home/u1527145/projects/medquant && bash slurm/launch_eval.sh"

.PHONY: monitor
monitor:
	watch -n 30 'ssh notchpeak "squeue -u u1527145"'

# ── Results ───────────────────────────────────────────────────────────────────

.PHONY: merge
merge:
	source medq_env/bin/activate && python src/eval/merge_results.py --config config/gpu_config.yaml

.PHONY: charts
charts:
	source medq_env/bin/activate && python src/analysis/charts.py --config config/gpu_config.yaml

# ── Utilities ─────────────────────────────────────────────────────────────────

.PHONY: clean-logs
clean-logs:
	rm -f slurm/logs/*.out slurm/logs/*.err

.PHONY: help
help:
	@echo "Local:        make test | test-cov | train-dry"
	@echo "Sync:         make sync-up | sync-results"
	@echo "Jobs:         make submit-train | submit-convert | submit-eval | monitor"
	@echo "Results:      make merge | charts"
	@echo "Utilities:    make clean-logs"
