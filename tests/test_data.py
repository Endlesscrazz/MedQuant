"""Tests for src/data — loader, formatter, contamination checks."""

import sys
from unittest.mock import patch

import pytest

from src.data.contamination_check import (
    check_medmcqa_overlap,
    check_pubmedqa_contamination,
)
from src.data.formatter import format_dataset, format_example
from src.data.loader import load_dataset


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeDataset:
    """Minimal HuggingFace Dataset stand-in for mocking _hf_load_dataset."""

    def __init__(self, items: list[dict]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def filter(self, fn) -> "FakeDataset":
        return FakeDataset([x for x in self._items if fn(x)])


class MockTokenizer:
    """Tokenizer mock that produces deterministic, inspectable output."""

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


def _fake_pubmedqa_row(pubid: int = 1, decision: str = "yes") -> dict:
    return {
        "pubid": pubid,
        "question": "Is ibuprofen an NSAID?",
        "context": {"contexts": ["Ibuprofen is commonly used.", "It reduces inflammation."], "labels": [], "meshes": []},
        "final_decision": decision,
        "long_answer": "Yes, ibuprofen is an NSAID.",
    }


def _fake_medmcqa_row(cop: int = 0, choice_type: str = "single") -> dict:
    return {
        "question": "What is the drug of choice for malaria?",
        "opa": "Chloroquine",
        "opb": "Amoxicillin",
        "opc": "Metformin",
        "opd": "Aspirin",
        "cop": cop,
        "choice_type": choice_type,
        "exp": "Chloroquine is used for malaria.",
        "subject_name": "Pharmacology",
        "topic_name": "Antimalarials",
    }


# ---------------------------------------------------------------------------
# 1. test_pubmedqa_loader
# ---------------------------------------------------------------------------


def test_pubmedqa_loader():
    rows = [_fake_pubmedqa_row(pubid=42, decision="no")]
    with patch("src.data.loader._hf_load_dataset", return_value=FakeDataset(rows)):
        items = load_dataset("pubmedqa", "train", cache_dir="/tmp")

    assert len(items) == 1
    item = items[0]
    assert item["task"] == "pubmedqa"
    assert item["pubid"] == 42
    assert item["question"] == "Is ibuprofen an NSAID?"
    assert "Ibuprofen is commonly used." in item["context"]
    assert "It reduces inflammation." in item["context"]
    assert item["answer"] == "no"


# ---------------------------------------------------------------------------
# 2. test_medmcqa_loader — choice_type filter
# ---------------------------------------------------------------------------


def test_medmcqa_loader():
    rows = [
        _fake_medmcqa_row(cop=0, choice_type="single"),
        _fake_medmcqa_row(cop=1, choice_type="multi"),   # must be filtered out
    ]
    with patch("src.data.loader._hf_load_dataset", return_value=FakeDataset(rows)):
        items = load_dataset("medmcqa", "train", cache_dir="/tmp")

    assert len(items) == 1
    item = items[0]
    assert item["task"] == "medmcqa"
    assert item["answer_idx"] == 0
    assert item["opa"] == "Chloroquine"
    assert "question" in item
    assert "opa" in item and "opb" in item and "opc" in item and "opd" in item


# ---------------------------------------------------------------------------
# 3. test_formatter_pubmedqa
# ---------------------------------------------------------------------------


def test_formatter_pubmedqa():
    item = {
        "pubid": 1,
        "question": "Does aspirin reduce fever?",
        "context": "Aspirin is an analgesic.",
        "answer": "yes",
        "task": "pubmedqa",
    }
    tok = MockTokenizer()
    result = format_example(item, tok, add_generation_prompt=False)

    assert "Aspirin is an analgesic." in result
    assert "Does aspirin reduce fever?" in result
    assert "yes, no, or maybe" in result


# ---------------------------------------------------------------------------
# 4. test_formatter_medmcqa
# ---------------------------------------------------------------------------


def test_formatter_medmcqa():
    item = {
        "question": "What treats malaria?",
        "opa": "Chloroquine",
        "opb": "Amoxicillin",
        "opc": "Metformin",
        "opd": "Aspirin",
        "answer_idx": 0,
        "task": "medmcqa",
    }
    tok = MockTokenizer()
    result = format_example(item, tok, add_generation_prompt=False)

    assert "What treats malaria?" in result
    assert "A) Chloroquine" in result
    assert "B) Amoxicillin" in result
    assert "C) Metformin" in result
    assert "D) Aspirin" in result
    assert "A, B, C, or D" in result


# ---------------------------------------------------------------------------
# 5. test_add_generation_prompt_false
# ---------------------------------------------------------------------------


def test_add_generation_prompt_false():
    """Training format must include the answer content, not a naked generation marker."""
    item = {
        "pubid": 1,
        "question": "Is X safe?",
        "context": "X has been studied.",
        "answer": "yes",
        "task": "pubmedqa",
    }
    tok = MockTokenizer()
    train_fmt = format_example(item, tok, add_generation_prompt=False)
    eval_fmt = format_example(item, tok, add_generation_prompt=True)

    # Training: answer is embedded inside an assistant turn, not a trailing naked marker
    assert "<assistant>yes</assistant>" in train_fmt
    assert not train_fmt.endswith("<assistant>")

    # Eval: ends with the generation prompt cue, no answer content
    assert eval_fmt.endswith("<assistant>")
    assert "yes</assistant>" not in eval_fmt


# ---------------------------------------------------------------------------
# 6. test_max_samples
# ---------------------------------------------------------------------------


def test_max_samples():
    rows = [_fake_pubmedqa_row(pubid=i) for i in range(20)]
    with patch("src.data.loader._hf_load_dataset", return_value=FakeDataset(rows)):
        items = load_dataset("pubmedqa", "train", cache_dir="/tmp", max_samples=5)

    assert len(items) == 5
    # Verify seed=42 reproducibility: same call returns same subset
    with patch("src.data.loader._hf_load_dataset", return_value=FakeDataset(rows)):
        items2 = load_dataset("pubmedqa", "train", cache_dir="/tmp", max_samples=5)
    assert [i["pubid"] for i in items] == [i["pubid"] for i in items2]


# ---------------------------------------------------------------------------
# 7. test_loader_raises_on_empty
# ---------------------------------------------------------------------------


def test_loader_raises_on_empty_pubmedqa():
    with patch("src.data.loader._hf_load_dataset", return_value=FakeDataset([])):
        with pytest.raises(ValueError, match="0 samples"):
            load_dataset("pubmedqa", "train", cache_dir="/tmp")


def test_loader_raises_on_empty_medmcqa_after_filter():
    """All rows are multi-choice; filter removes all → ValueError."""
    rows = [_fake_medmcqa_row(choice_type="multi") for _ in range(3)]
    with patch("src.data.loader._hf_load_dataset", return_value=FakeDataset(rows)):
        with pytest.raises(ValueError, match="0 samples"):
            load_dataset("medmcqa", "train", cache_dir="/tmp")


# ---------------------------------------------------------------------------
# 8. test_pubmedqa_contamination
# ---------------------------------------------------------------------------


def test_pubmedqa_contamination():
    train = [
        {"pubid": 1, "question": "Q1", "context": "C1", "answer": "yes", "task": "pubmedqa"},
        {"pubid": 2, "question": "Q2", "context": "C2", "answer": "no", "task": "pubmedqa"},
        {"pubid": 3, "question": "Q3", "context": "C3", "answer": "maybe", "task": "pubmedqa"},
    ]
    # pubid 2 appears in eval → should be removed
    eval_items = [
        {"pubid": 2, "question": "Q2", "context": "C2", "answer": "no", "task": "pubmedqa"},
    ]

    cleaned, n_removed = check_pubmedqa_contamination(train, eval_items)

    assert n_removed == 1
    assert len(cleaned) == 2
    assert all(item["pubid"] != 2 for item in cleaned)


def test_pubmedqa_contamination_zero_overlap():
    train = [{"pubid": i, "question": "Q", "context": "C", "answer": "yes", "task": "pubmedqa"} for i in range(5)]
    eval_items = [{"pubid": 99, "question": "Q", "context": "C", "answer": "yes", "task": "pubmedqa"}]

    cleaned, n_removed = check_pubmedqa_contamination(train, eval_items)

    assert n_removed == 0
    assert len(cleaned) == 5


# ---------------------------------------------------------------------------
# 9. test_medmcqa_overlap
# ---------------------------------------------------------------------------


def test_medmcqa_overlap_found():
    train = [{"question": "What treats malaria?", "task": "medmcqa"}]
    val = [{"question": "  What treats malaria?  ", "task": "medmcqa"}]  # extra whitespace

    overlap = check_medmcqa_overlap(train, val)
    assert overlap == 1


def test_medmcqa_overlap_none():
    train = [{"question": "What is aspirin?", "task": "medmcqa"}]
    val = [{"question": "What is penicillin?", "task": "medmcqa"}]

    overlap = check_medmcqa_overlap(train, val)
    assert overlap == 0
