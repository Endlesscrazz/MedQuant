"""Contamination checks between training and evaluation splits."""

import hashlib
import sys


def check_pubmedqa_contamination(
    train_items: list[dict],
    eval_items: list[dict],
) -> tuple[list[dict], int]:
    """Remove pqa_artificial train items whose pubid appears in the pqa_labeled eval set.

    Expected n_removed is 0 — the two configs share no PubMed IDs in practice,
    but the check is mandatory for reproducibility.
    """
    eval_pubids = {item["pubid"] for item in eval_items}
    cleaned = [item for item in train_items if item["pubid"] not in eval_pubids]
    n_removed = len(train_items) - len(cleaned)
    print(
        f"[contamination] PubMedQA contamination check: removed {n_removed} samples",
        file=sys.stderr,
    )
    return cleaned, n_removed


def _question_hash(text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def check_medmcqa_overlap(
    train_items: list[dict],
    val_items: list[dict],
) -> int:
    """Count train questions whose normalized text hash matches any validation question.

    Expected count is 0. Logs result to stderr.
    """
    val_hashes = {_question_hash(item["question"]) for item in val_items}
    overlap = sum(
        1 for item in train_items if _question_hash(item["question"]) in val_hashes
    )
    print(
        f"[contamination] MedMCQA overlap check: {overlap} overlapping questions",
        file=sys.stderr,
    )
    return overlap
