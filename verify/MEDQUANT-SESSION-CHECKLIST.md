# Context-Bridge — MedQuant Session Checklist
# Written: 2026-04-28
# Purpose: exact steps to follow after every MedQuant session to generate
#          one labeled eval pair for CB Gate E. Keep this open in a second
#          window while working on MedQuant.
# ──────────────────────────────────────────────────────────────────────────

## ONE-TIME SETUP (do before Session 1 only)

```bash
cd ~/Desktop/UoU/Claude-workspace/projects/MedQuant
source ~/Desktop/UoU/Claude-workspace/projects/Context-Bridge/cb_env/bin/activate
context-bridge install    # already done — just verify .mcp.json exists
```

---

## AT THE START OF EVERY SESSION (inside VS Code)

Type this in the Claude Code chat before saying anything else:

```
/bridge
```

Claude calls `get_resume_packet`, reads the summary of the last session,
and orients itself. You can continue immediately without re-explaining context.
Session 1: CB returns a first_run packet — expected, nothing is wrong.

---

## AT THE END OF EVERY SESSION (while Claude Code is still open)

Do this BEFORE closing the chat. Claude still has full context of what just happened.

### End-of-session step — Get the ground truth from Claude

Ask Claude in the chat:

```
Give me the session summary
```

Claude outputs the DONE block (per the CLAUDE.md format). It looks like:

```
✅ DONE: Session 1

WHAT WAS BUILT: ...
KEY DECISION:   ...
TRADE-OFF:      ...

Next: ...
```

Then run this in your terminal (VS Code integrated terminal is fine):

```bash
git diff --stat HEAD
```

This shows every file that was modified this session. No memory needed.

Now create `verify/ground_truth/session_N.json` by copying from those two sources:

```json
{
  "session": 1,
  "date": "2026-MM-DD",
  "files_touched": ["<copy from git diff --stat output>"],
  "artifacts_created": [],
  "chpc_job_ids": [],
  "scratch_paths": [],
  "key_decision": "<copy KEY DECISION line from Claude's session summary>",
  "last_completed": "<copy WHAT WAS BUILT from Claude's session summary>",
  "next_step": "<copy Next: line from Claude's session summary>",
  "blockers": [],
  "continuity_level_expected": "assistant_derived"
}
```

Field notes:
- `files_touched`: paste the file list from `git diff --stat`
- `artifacts_created`: CHPC outputs this session (checkpoints, GGUFs) — leave `[]` if none
- `chpc_job_ids`: SLURM job IDs submitted or completed — leave `[]` if none
- `scratch_paths`: CHPC scratch paths now meaningful — leave `[]` if none
- `blockers`: leave `[]` if session completed cleanly
- `continuity_level_expected`: `first_run` for Session 1, `assistant_derived` for 2–5

This takes about 1 minute. Everything comes from what's already on your screen.

Now close the chat or start a new one. The SessionEnd hook fires automatically.

---

## AFTER THE SESSION (from Mac Terminal, ~4 minutes)

```bash
source ~/Desktop/UoU/Claude-workspace/projects/Context-Bridge/cb_env/bin/activate
cd ~/Desktop/UoU/Claude-workspace/projects/MedQuant
```

### Step 1 — Verify CB captured the session

```bash
context-bridge simulate .
context-bridge log --tail 1
```

What to look for:
- Session 1: `continuity_level: first_run` — expected
- Sessions 2+: `continuity_level: assistant_derived`
- `files[]` should include files from your ground truth
- `next_step` should roughly match what you wrote

If packet is empty: wait 10 seconds and retry. If still empty, CB will catch
it automatically next session — continue to Step 2 anyway.

### Step 2 — Score the packet

```bash
python verify/eval_cb_packet.py verify/ground_truth/session_N.json
```

Prints score 0–4 and field-by-field pass/fail.
Also writes `verify/ground_truth/session_N_eval.json`.

Score guide:
- 3–4 → CB captured the session well
- 1–2 → CB missed something — note what in Step 3
- 0 → hook likely didn't fire, check CB install

### Step 3 — Grade Q1 and Q2

Open `~/Desktop/UoU/Claude-workspace/projects/Context-Bridge/metrics/dogfood_log.md`
and append:

```
## MedQuant Session N — YYYY-MM-DD
CB score: X/4 | continuity_level returned: ___

Q1 — At the start of THIS session, when you typed /bridge, could Claude
     resume without asking to read any files?
  Grade: ___ / 5
  Reason: ___

Q2 — Was anything in the CB packet wrong or made up?
  Grade: ___ / 5
  Reason: ___

What CB got right: ___
What CB missed: ___
```

Q1 grading (grade retrospectively — based on how /bridge worked at START of this session):
  5 — Claude resumed immediately, asked for no file reads
  4 — Mostly fine, one minor clarification needed
  3 — Good orientation but still needed 1–2 file reads
  2 — Partially useful, needed significant re-reading
  1 — Not useful, had to explain everything from scratch

Q2 grading (based on CB packet you saw in Step 1):
  5 — Everything accurate
  4 — Minor imprecision, nothing wrong
  3 — One field wrong or misleading
  2 — Multiple fields wrong
  1 — Hallucinated or fabricated content

NOTE: You cannot grade Q1 for Session 1 — you had no prior session.
Leave Q1 blank for Session 1. Grade Q1 from Session 2 onward.

### Step 4 — Record feedback signal

```bash
# If CB did well:
context-bridge feedback --good --note "brief note on what worked"

# If CB missed something:
context-bridge feedback --bad --note "what was missing: ___"
```

---

## SESSION TRACKING TABLE

| Session | Date | CB Score | Q1 | Q2 | Notes |
|---|---|---|---|---|---|
| 1 | | | n/a | | first_run expected, no prior session to grade |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| **Avg** | | | | | Target: Q1 ≥ 3.0, Q2 ≥ 4.0 (Sessions 2–5) |

Gate E closes when Q1 avg ≥ 3.0 AND Q2 avg ≥ 4.0 across Sessions 2–5.
