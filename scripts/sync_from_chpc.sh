#!/bin/bash
# Pull results (metrics JSON, logs) from CHPC scratch back to local.
# Never pulls model weights or GGUFs.
# Usage: bash scripts/sync_from_chpc.sh

set -euo pipefail

SCRATCH_REMOTE=notchpeak:/scratch/general/vast/u1527145/medquant/
PROJECT_REMOTE=notchpeak:/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant/
LOCAL=/Users/shreyas/Desktop/UoU/Claude-workspace/projects/MedQuant/

echo "Pulling metrics and logs from CHPC..."

# Pull per-variant raw eval results
rsync -avz --progress \
    --include='*.json' \
    --exclude='*' \
    "${SCRATCH_REMOTE}metrics/raw/" \
    "${LOCAL}metrics/raw/"

# Pull merged results if present
rsync -avz --progress \
    "${SCRATCH_REMOTE}metrics/results.json" \
    "${LOCAL}metrics/" 2>/dev/null || echo "(results.json not yet on scratch — skipping)"

# Pull SLURM logs (last run only — overwrite)
rsync -avz --progress \
    --include='*.out' \
    --include='*.err' \
    --exclude='*' \
    "${SCRATCH_REMOTE}logs/" \
    "${LOCAL}slurm/logs/" 2>/dev/null || true

echo "Done. Check metrics/raw/ for per-variant JSON files."
