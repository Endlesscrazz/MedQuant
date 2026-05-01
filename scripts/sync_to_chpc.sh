#!/bin/bash
# Push local code changes to CHPC.
# Excludes model weights, venv, and scratch artifacts.
# Usage: bash scripts/sync_to_chpc.sh

set -euo pipefail

LOCAL=/Users/shreyas/Desktop/UoU/Claude-workspace/projects/MedQuant/
REMOTE=notchpeak:/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant/

echo "Syncing code → CHPC..."

rsync -avz --progress \
    --exclude='.git/' \
    --exclude='medq_env/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.gguf' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='outputs/' \
    --exclude='metrics/raw/*.json' \
    --exclude='metrics/charts/' \
    --exclude='slurm/logs/' \
    --exclude='.DS_Store' \
    --exclude='context-bridge-state.db' \
    --exclude='context-bridge-log.md' \
    "${LOCAL}" "${REMOTE}"

echo "Done. Run 'squeue -u u1527145' on CHPC to check job queue."
