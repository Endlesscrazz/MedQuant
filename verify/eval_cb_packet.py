"""
Diff a CB resume packet against a ground truth session label.

Usage:
    python verify/eval_cb_packet.py verify/ground_truth/session_1.json

Prints: score (0-4), field-by-field pass/fail, and the full CB packet.
Writes: verify/ground_truth/session_N_eval.json alongside the input file.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def run_simulate(project_dir: str) -> dict:
    env = os.environ.copy()
    env["PATH"] = (
        str(Path.home() / "Desktop/UoU/Claude-workspace/projects/Context-Bridge/cb_env/bin")
        + ":" + env.get("PATH", "")
    )
    result = subprocess.run(
        ["context-bridge", "simulate", project_dir, "--json"],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(f"[eval_cb_packet] simulate failed: {result.stderr.strip()}", file=sys.stderr)
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"[eval_cb_packet] could not parse simulate output", file=sys.stderr)
        return {}


def word_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def score_packet(packet: dict, truth: dict) -> dict:
    results = {}

    # 1. files_touched: any ground truth file appears in packet files[]
    packet_files = [f.get("path", "") if isinstance(f, dict) else str(f)
                    for f in packet.get("files", [])]
    truth_files = truth.get("files_touched", [])
    files_hit = any(
        any(tf in pf or pf in tf for pf in packet_files)
        for tf in truth_files
    )
    results["files_touched"] = {
        "pass": files_hit,
        "truth": truth_files,
        "packet": packet_files,
    }

    # 2. key_decision: substring found in packet summary or decisions
    key_decision = truth.get("key_decision", "").lower()
    summary = (packet.get("last_session", {}).get("summary", "") or "").lower()
    decisions = packet.get("last_session", {}).get("decisions", []) or []
    decisions_text = " ".join(
        (d.get("text", "") if isinstance(d, dict) else str(d)).lower()
        for d in decisions
    )
    decision_hit = key_decision and (
        key_decision in summary or key_decision in decisions_text
        or any(w in summary + " " + decisions_text for w in key_decision.split() if len(w) > 5)
    )
    results["key_decision"] = {
        "pass": bool(decision_hit),
        "truth": truth.get("key_decision", ""),
        "packet_summary": packet.get("last_session", {}).get("summary", ""),
    }

    # 3. next_step: >=50% word overlap
    truth_next = truth.get("next_step", "")
    packet_next = packet.get("last_session", {}).get("next_step", "") or ""
    overlap = word_overlap(truth_next, packet_next)
    results["next_step"] = {
        "pass": overlap >= 0.5,
        "overlap": round(overlap, 2),
        "truth": truth_next,
        "packet": packet_next,
    }

    # 4. continuity_level: exact match
    expected_level = truth.get("continuity_level_expected", "")
    actual_level = packet.get("continuity_level", "")
    results["continuity_level"] = {
        "pass": expected_level == actual_level,
        "expected": expected_level,
        "actual": actual_level,
    }

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify/eval_cb_packet.py verify/ground_truth/session_N.json")
        sys.exit(1)

    truth_path = sys.argv[1]
    with open(truth_path) as f:
        truth = json.load(f)

    project_dir = str(Path(__file__).parent.parent)
    print(f"Running context-bridge simulate on: {project_dir}")
    packet = run_simulate(project_dir)

    if not packet:
        print("ERROR: could not get CB packet. Is context-bridge installed and active?")
        sys.exit(1)

    scores = score_packet(packet, truth)
    total = sum(1 for v in scores.values() if v["pass"])

    print(f"\n{'='*50}")
    print(f"SCORE: {total}/4  (Session {truth.get('session', '?')})")
    print(f"{'='*50}")

    for field, result in scores.items():
        status = "PASS" if result["pass"] else "FAIL"
        print(f"\n[{status}] {field}")
        for k, v in result.items():
            if k != "pass":
                print(f"       {k}: {v}")

    print(f"\n{'='*50}")
    print("FULL PACKET:")
    print(json.dumps(packet, indent=2))

    # Write eval result
    eval_path = truth_path.replace(".json", "_eval.json")
    eval_result = {
        "session": truth.get("session"),
        "date": truth.get("date"),
        "score": total,
        "max_score": 4,
        "fields": scores,
        "packet": packet,
    }
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nEval result written to: {eval_path}")


if __name__ == "__main__":
    main()
