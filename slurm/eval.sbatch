#!/bin/bash
#SBATCH --job-name=medquant-eval
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/general/vast/u1527145/medquant/logs/eval_%j_%a.out
#SBATCH --error=/scratch/general/vast/u1527145/medquant/logs/eval_%j_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shreyas.patiledu07@gmail.com

# One job per model variant. SLURM_ARRAY_TASK_ID selects the variant.
# Submit via launch_eval.sh — do not submit this directly.

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
module load cuda/12.1

SCRATCH=/scratch/general/vast/u1527145/medquant
PROJECT=/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant

export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ${PROJECT}/medq_env/bin/activate

mkdir -p ${SCRATCH}/logs

# ── Variant selection ─────────────────────────────────────────────────────────
# 10 variants: base × 5 quant levels + finetuned × 5 quant levels
VARIANTS=(
    "base_f16"
    "base_q8_0"
    "base_q6_k"
    "base_q4_k_m"
    "base_q2_k"
    "finetuned_f16"
    "finetuned_q8_0"
    "finetuned_q6_k"
    "finetuned_q4_k_m"
    "finetuned_q2_k"
)

VARIANT=${VARIANTS[${SLURM_ARRAY_TASK_ID}]}

echo "Job ID:  ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Variant: ${VARIANT}"

# ── Eval ──────────────────────────────────────────────────────────────────────
cd ${PROJECT}

python src/eval/batch_eval.py \
    --config config/gpu_config.yaml \
    --variant "${VARIANT}" \
    --output "metrics/raw/${VARIANT}.json"

echo "Eval complete for ${VARIANT}. Job done."
