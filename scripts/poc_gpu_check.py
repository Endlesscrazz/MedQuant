"""
PoC: verify GPU environment on CHPC before Session 1.
Writes a JSON result to stdout and to --output path if given.
Run via slurm/poc_check.sbatch — not intended for local execution.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unavailable"


def check_torch():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_ok else 0
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "none"
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if device_count > 0 else 0
        # Quick tensor op on GPU
        if cuda_ok:
            x = torch.randn(1000, 1000, device="cuda")
            _ = (x @ x).sum().item()
            gpu_compute = "ok"
        else:
            gpu_compute = "no CUDA"
        return {
            "torch_version": torch.__version__,
            "cuda_available": cuda_ok,
            "device_count": device_count,
            "device_name": device_name,
            "vram_gb": vram_gb,
            "gpu_compute": gpu_compute,
        }
    except ImportError:
        return {"error": "torch not installed"}


def check_transformers():
    try:
        import transformers
        return {"transformers_version": transformers.__version__}
    except ImportError:
        return {"error": "transformers not installed"}


def check_peft():
    try:
        import peft
        return {"peft_version": peft.__version__}
    except ImportError:
        return {"error": "peft not installed"}


def check_trl():
    try:
        import trl
        return {"trl_version": trl.__version__}
    except ImportError:
        return {"error": "trl not installed"}


def check_bitsandbytes():
    try:
        import bitsandbytes as bnb
        return {"bitsandbytes_version": bnb.__version__}
    except ImportError:
        return {"error": "bitsandbytes not installed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Path to write JSON result")
    args = parser.parse_args()

    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_node": os.environ.get("SLURM_NODELIST", "local"),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                           "--format=csv,noheader"]),
        **check_torch(),
        **check_transformers(),
        **check_peft(),
        **check_trl(),
        **check_bitsandbytes(),
    }

    # Pass / fail summary
    checks = {
        "cuda": result.get("cuda_available", False),
        "gpu_compute": result.get("gpu_compute") == "ok",
        "torch": "error" not in result.get("torch_version", "error"),
        "transformers": "error" not in result.get("transformers_version", "error"),
        "peft": "error" not in result.get("peft_version", "error"),
        "trl": "error" not in result.get("trl_version", "error"),
        "bitsandbytes": "error" not in result.get("bitsandbytes_version", "error"),
    }
    result["checks"] = checks
    result["all_pass"] = all(checks.values())

    output_json = json.dumps(result, indent=2)
    print(output_json)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nResult written to: {args.output}", file=sys.stderr)

    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
