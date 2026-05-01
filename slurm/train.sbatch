#!/bin/bash
#SBATCH --job-name=medquant-train
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/general/vast/u1527145/medquant/logs/train_%j.out
#SBATCH --error=/scratch/general/vast/u1527145/medquant/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shreyas.patiledu07@gmail.com

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
module load cuda/12.1

SCRATCH=/scratch/general/vast/u1527145/medquant
PROJECT=/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant

export HF_HOME=${SCRATCH}/hf_cache
export HF_HUB_CACHE=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ${PROJECT}/medq_env/bin/activate

mkdir -p ${SCRATCH}/logs

# ── Parse args ────────────────────────────────────────────────────────────────
CONFIG=${1:-${PROJECT}/config/gpu_config.yaml}
DRY_RUN=${2:-}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG}"

# ── Train ─────────────────────────────────────────────────────────────────────
cd ${PROJECT}

python src/train/train.py \
    --config "${CONFIG}" \
    ${DRY_RUN:+--dry-run}

echo "Training complete. Job ${SLURM_JOB_ID} done."
