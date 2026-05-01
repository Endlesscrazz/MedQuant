"""Load PubMedQA and MedMCQA from HuggingFace and return standardized dicts.

Offline mode: set TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1, HF_HUB_OFFLINE=1
in the environment before running (handled by SLURM job scripts on CHPC).
"""

import random
import sys
from typing import Literal

from datasets import load_dataset as _hf_load_dataset

# PubMedQA: train split uses pqa_artificial, validation uses pqa_labeled.
# pqa_labeled has NO "test" split — calling split="test" raises KeyError.
_PUBMEDQA_SPLIT_MAP = {
    "train": ("pqa_artificial", "train"),
    "validation": ("pqa_labeled", "train"),
}

_REQUIRED_PUBMEDQA = {"pubid", "question", "context", "final_decision"}
_REQUIRED_MEDMCQA = {"question", "opa", "opb", "opc", "opd", "cop", "choice_type"}


def _standardize_pubmedqa(row: dict) -> dict:
    context_str = "\n\n".join(row["context"]["contexts"])
    return {
        "pubid": row["pubid"],
        "question": row["question"],
        "context": context_str,
        "answer": row["final_decision"],
        "task": "pubmedqa",
    }


def _standardize_medmcqa(row: dict) -> dict:
    return {
        "question": row["question"],
        "opa": row["opa"],
        "opb": row["opb"],
        "opc": row["opc"],
        "opd": row["opd"],
        "answer_idx": int(row["cop"]),
        "task": "medmcqa",
    }


def load_dataset(
    name: Literal["pubmedqa", "medmcqa"],
    split: Literal["train", "validation", "test"],
    cache_dir: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Load and standardize a dataset split.

    Returns:
        pubmedqa: list of {pubid, question, context, answer, task}
        medmcqa:  list of {question, opa, opb, opc, opd, answer_idx, task}
    Raises:
        ValueError: empty result or schema mismatch
        RuntimeError: HuggingFace load failure
    """
    try:
        if name == "pubmedqa":
            if split not in _PUBMEDQA_SPLIT_MAP:
                raise ValueError(
                    f"PubMedQA split must be 'train' or 'validation', got '{split}'"
                )
            hf_config, hf_split = _PUBMEDQA_SPLIT_MAP[split]
            raw = _hf_load_dataset(
                "qiaojin/PubMedQA", hf_config, split=hf_split, cache_dir=cache_dir
            )
            if len(raw) == 0:
                raise ValueError(f"PubMedQA/{split} returned 0 samples before filtering.")
            missing = _REQUIRED_PUBMEDQA - set(raw[0].keys())
            if missing:
                raise ValueError(f"PubMedQA schema mismatch: missing fields {missing}")
            items = [_standardize_pubmedqa(r) for r in raw]

        elif name == "medmcqa":
            raw = _hf_load_dataset("medmcqa", split=split, cache_dir=cache_dir)
            if len(raw) == 0:
                raise ValueError(f"MedMCQA/{split} returned 0 samples before filtering.")
            missing = _REQUIRED_MEDMCQA - set(raw[0].keys())
            if missing:
                raise ValueError(f"MedMCQA schema mismatch: missing fields {missing}")
            raw = raw.filter(lambda x: x["choice_type"] == "single")
            items = [_standardize_medmcqa(r) for r in raw]

        else:
            raise ValueError(f"Unknown dataset '{name}'. Use 'pubmedqa' or 'medmcqa'.")

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load '{name}/{split}' from HuggingFace: {e}") from e

    if not items:
        raise ValueError(
            f"Dataset '{name}/{split}' returned 0 samples after filtering."
        )

    if max_samples is not None and len(items) > max_samples:
        rng = random.Random(42)
        rng.shuffle(items)
        items = items[:max_samples]

    print(f"[loader] {name}/{split}: {len(items)} samples", file=sys.stderr)
    return items


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Smoke-test dataset loading")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["hf_cache_dir"]
    max_samples = cfg.get("max_samples_per_dataset")

    try:
        load_dataset("pubmedqa", "train", cache_dir, max_samples)
        load_dataset("pubmedqa", "validation", cache_dir, max_samples)
        load_dataset("medmcqa", "train", cache_dir, max_samples)
        load_dataset("medmcqa", "validation", cache_dir, max_samples)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
