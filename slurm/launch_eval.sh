#!/bin/bash
# Submit all 10 eval jobs as a SLURM array.
# Run from project root on the CHPC login node.
# Usage: bash slurm/launch_eval.sh

set -euo pipefail

PROJECT=/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant

echo "Submitting eval array (10 variants)..."

JOB_ID=$(sbatch \
    --array=0-9 \
    ${PROJECT}/slurm/eval.sbatch \
    | awk '{print $4}')

echo "Submitted array job: ${JOB_ID}"
echo "Monitor with:  squeue -u u1527145"
echo "Logs at:       /scratch/general/vast/u1527145/medquant/logs/eval_${JOB_ID}_*.out"
echo ""
echo "After all jobs complete, merge results:"
echo "  cd ${PROJECT} && python src/eval/merge_results.py --config config/gpu_config.yaml"
