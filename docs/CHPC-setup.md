# MedQuant — CHPC Setup Guide
# Written: 2026-04-27
# Purpose: step-by-step environment setup on University of Utah CHPC,
#          HuggingFace model download, llama.cpp build, SLURM usage.
# Cluster: notchpeak / CHPC VAST storage, SLURM scheduler
# ──────────────────────────────────────────────────────────────────────────

## OVERVIEW

CHPC is only used for:
1. Downloading model weights and datasets (login node)
2. Running SLURM training and eval jobs (compute nodes)
3. Storing large artifacts on scratch (weights, GGUF files)

All development happens locally on Mac. Sync via git.

---

## STEP 1 — SSH ACCESS

Login node:
```bash
ssh u1527145@notchpeak.chpc.utah.edu
# or
ssh u1527145@lonepeak.chpc.utah.edu
```

Recommended: add to ~/.ssh/config locally
```
Host chpc
    HostName notchpeak.chpc.utah.edu
    User u1527145
    IdentityFile ~/.ssh/id_rsa
```
Then: ssh chpc

---

## STEP 2 — SCRATCH DIRECTORY STRUCTURE

Create the scratch layout before downloading anything:
```bash
SCRATCH=/scratch/general/vast/u1527145/medquant
mkdir -p $SCRATCH/models/llama-3.1-8b
mkdir -p $SCRATCH/models/base-gguf
mkdir -p $SCRATCH/models/finetuned-gguf
mkdir -p $SCRATCH/hf_cache
mkdir -p $SCRATCH/datasets
mkdir -p $SCRATCH/outputs/checkpoints
mkdir -p $SCRATCH/outputs/logs
mkdir -p $SCRATCH/outputs/merged
```

Add to ~/.bashrc on CHPC for convenience:
```bash
export SCRATCH=/scratch/general/vast/u1527145/medquant
export HF_HOME=$SCRATCH/hf_cache
```

---

## STEP 3 — PYTHON ENVIRONMENT

Check available modules first:
```bash
module spider python
module spider cuda
# Note the exact version strings shown — use them in the commands below.
# Do NOT copy cuda/12.x verbatim; substitute the actual version from the output.
```

Load modules and create environment:
```bash
module load python/3.11         # substitute actual version from module spider
module load cuda/12.1           # substitute actual CUDA version from module spider

python -m venv ~/envs/medquant
source ~/envs/medquant/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft bitsandbytes trl accelerate
pip install sentencepiece protobuf huggingface_hub
# llama-cpp-python MUST be built with CUDA support for GPU eval:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install pytest pyyaml matplotlib fastapi uvicorn
```

Verify GPU is visible (login nodes have no GPU — must use an interactive compute node):
```bash
# Request a short interactive GPU allocation:
salloc --account=yqu-gpu-np --partition=yqu-gpu-np --gpus=1 --time=00:30:00

# Once on the compute node, activate the venv and check:
source ~/envs/medquant/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True, NVIDIA A800 80GB or similar

# Verify llama-cpp-python can offload to GPU (not just import):
# Run this after Session 3 produces the first GGUF. If no GGUF yet,
# skip and revisit at Session 4.0 logprob spike.
# python -c "
# from llama_cpp import Llama
# m = Llama('/path/to/any-small.gguf', n_gpu_layers=-1, verbose=True)
# print(m('Hello', max_tokens=3))
# "
# Look for CUDA/cuBLAS in the verbose output. CPU-only shows 'GGML_OPENCL' or nothing.
# A CPU-only install silently distorts latency — GPU offload must be confirmed.

# Exit the interactive session after verifying:
exit
```

Add venv activation to ~/.bashrc:
```bash
source ~/envs/medquant/bin/activate
```

---

## STEP 4 — HUGGINGFACE AUTHENTICATION

Llama-3.1-8B is a gated model. You need to:
1. Accept the Meta license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Create a HuggingFace access token at https://huggingface.co/settings/tokens

On CHPC login node:
```bash
huggingface-cli login
# Paste your token when prompted
# This saves to ~/.cache/huggingface/token
```

Alternatively set env var:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
# Add to ~/.bashrc
```

---

## STEP 5 — DOWNLOAD MODELS AND DATASETS

Run ALL downloads on the login node (has internet). Compute nodes do not.

### Download Llama-3.1-8B-Instruct
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir $SCRATCH/models/llama-3.1-8b \
    --token $HF_TOKEN
```
Expected size: ~16GB. Takes 10–20 min depending on network speed.

### Download datasets (cache to HF_HOME)
```bash
export HF_HOME=$SCRATCH/hf_cache
python -c "
from datasets import load_dataset
# PubMedQA — pqa_artificial for training
ds = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
print('PubMedQA artificial:', len(ds))
# PubMedQA — pqa_labeled for eval (CRITICAL: split='train', NOT 'test')
# Calling split='test' here raises KeyError — pqa_labeled has no test split.
ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
print('PubMedQA labeled:', len(ds))   # Expected: 1000
# MedMCQA
ds = load_dataset('openlifescienceai/medmcqa', split='train')
print('MedMCQA train:', len(ds))       # Expected: ~182K after filter
ds = load_dataset('openlifescienceai/medmcqa', split='validation')
print('MedMCQA validation:', len(ds))  # Expected: 4183
"
```
Datasets are small (~1GB total). Downloads quickly.

### Verify downloads
```bash
ls -lh $SCRATCH/models/llama-3.1-8b/
# Should show: config.json, tokenizer.json, model-0000x-of-0000y.safetensors, etc.
du -sh $SCRATCH/hf_cache/
# Should be ~1GB
```

---

## STEP 6 — CLONE PROJECT ON CHPC

```bash
cd /uufs/chpc.utah.edu/common/home/u1527145
mkdir -p projects && cd projects
git clone https://github.com/Endlesscrazz/medquant.git
cd medquant
```

Update config/gpu_config.yaml scratch paths to match your CHPC scratch:
```yaml
scratch_base: /scratch/general/vast/u1527145/medquant
hf_cache_dir: /scratch/general/vast/u1527145/medquant/hf_cache
```

---

## STEP 7 — BUILD llama.cpp

Clone and build on login node (this takes ~5 min):
```bash
cd /uufs/chpc.utah.edu/common/home/u1527145
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

module load cmake gcc cuda/12.1    # substitute actual version from: module spider cuda
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 8
```

Test build:
```bash
./build/bin/llama-cli --version
./build/bin/llama-quantize --help
```

Note the full path to llama.cpp for use in convert scripts:
```bash
echo $(pwd)
# /uufs/chpc.utah.edu/common/home/u1527145/llama.cpp
```

Update convert scripts / SLURM jobs to use this path.

---

## STEP 8 — TEST DRY-RUN BEFORE SUBMITTING TRAINING

On the login node (brief test — do not run full training on login node):
```bash
cd /uufs/chpc.utah.edu/common/home/u1527145/projects/medquant
source ~/envs/medquant/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python src/train/train.py --config config/gpu_config.yaml --dry-run
```

Expected output: batch shape, one forward pass loss value, then exit.
If this fails, debug here before submitting a SLURM job.

---

## STEP 9 — SUBMITTING AND MONITORING SLURM JOBS

Submit training:
```bash
cd /uufs/chpc.utah.edu/common/home/u1527145/projects/medquant
sbatch slurm/train.sbatch
```

Monitor:
```bash
squeue -u u1527145              # see your jobs
squeue -u u1527145 -l           # with details
scancel <JOBID>                 # cancel a job
```

View live job output:
```bash
tail -f outputs/logs/train_<JOBID>.out
```

Check job history / efficiency after completion:
```bash
seff <JOBID>
sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,MaxVMSize
```

---

## STEP 10 — SYNCING RESULTS BACK TO LOCAL

For small files (logs, metrics JSON):
```bash
# From local Mac:
rsync -avz chpc:/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant/outputs/logs/ \
    ~/Desktop/UoU/Claude-workspace/projects/MedQuant/outputs/logs/

rsync -avz chpc:/uufs/chpc.utah.edu/common/home/u1527145/projects/medquant/metrics/ \
    ~/Desktop/UoU/Claude-workspace/projects/MedQuant/metrics/
```

Or via git (if logs/metrics are small enough to commit):
```bash
# On CHPC
git add metrics/ outputs/logs/
git commit -m "training results session 2"
git push

# On local Mac
git pull
```

For GGUF files (large — rsync only, never git):
```bash
rsync -avz --progress chpc:$SCRATCH/models/finetuned-gguf/ \
    /scratch/local/medquant/gguf/
```

---

## COMMON ISSUES

**"CUDA out of memory" during training:**
- Reduce batch_size in gpu_config.yaml (try 16 or 8)
- Increase gradient_accumulation_steps to keep effective batch size at 32
- Enable gradient checkpointing if not already set

**"Model not found" on compute node:**
- Compute nodes have no internet. Set TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1 in sbatch script.
- Verify HF_HOME points to scratch where you downloaded the model.

**"Permission denied" on scratch:**
- CHPC scratch auto-purges files not accessed in 60–90 days.
- Re-download if needed. Keep important artifacts in home dir or re-download
  from git history with download scripts.

**bitsandbytes not finding CUDA:**
- Make sure `module load cuda` is in your sbatch before activating venv.
- Verify: python -c "import bitsandbytes as bnb; print(bnb.__version__)"

**llama.cpp build fails on CHPC:**
- Try without CUDA first: cmake -B build (CPU only)
- Then test quantize on a small model
- GPU layers (-ngl flag) can be set to 0 for CPU-only eval if needed
