#!/bin/bash
#SBATCH --job-name=medquant-poc
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/general/vast/u1527145/medquant/logs/poc_%j.out
#SBATCH --error=/scratch/general/vast/u1527145/medquant/logs/poc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shreyas.patiledu07@gmail.com

set -euo pipefail

module load cuda/12.1

SCRATCH=/scratch/general/vast/u1527145/medquant
PROJECT=/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant

mkdir -p ${SCRATCH}/logs ${SCRATCH}/poc

echo "=== MedQuant GPU PoC ==="
echo "Job:  ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================"

source ${PROJECT}/medq_env/bin/activate

python ${PROJECT}/scripts/poc_gpu_check.py \
    --output ${SCRATCH}/poc/gpu_check_${SLURM_JOB_ID}.json

echo "PoC complete. Exit code: $?"
