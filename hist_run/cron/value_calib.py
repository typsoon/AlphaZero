#!/usr/bin/env python3
"""Extract value-head calibration metrics from the last puzzle-eval results.json.

The puzzle evaluator records, per puzzle, the network's value output
(`network_value`, side-to-move perspective in [-1, +1]) alongside the puzzle
category. Calibration = does the value head agree with the ground-truth
outcome of the position?

  - "losing" categories (side-to-move is being mated): value SHOULD be ~ -1.
  - "winning" categories (side-to-move has a mate / wins material): value
    SHOULD be ~ +1.

At the current bootstrap stage the policy finds the tactic but the value head
under-values won positions, so the winning-category mean sits well below +1.
Tracking that mean climbing toward positive is the lead indicator that the
value head is actually calibrating (as opposed to the policy papering over it).

Appends one JSON line per call to value_calib_log.jsonl and prints a
RESULT_CALIB line for the calling loop.
"""

import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "performance_evaluation" / "results.json"
LOG = Path(__file__).resolve().parent / "value_calib_log.jsonl"

# Ground-truth sign of the side-to-move's position by puzzle category.
LOSING = {"avoid_mate_in_1"}
WINNING = {"mate_in_1", "mate_in_1_or_lose", "mate_in_2", "win_material"}


def mean(xs):
    return st.mean(xs) if xs else None


def main():
    if not RESULTS.exists():
        print("RESULT_CALIB: SKIP (no results.json)")
        return 0
    results = json.load(open(RESULTS))
    win_vals = [r["network_value"] for r in results if r["category"] in WINNING]
    lose_vals = [r["network_value"] for r in results if r["category"] in LOSING]
    m1_vals = [r["network_value"] for r in results if r["category"] == "mate_in_1"]

    win_mean = mean(win_vals)
    lose_mean = mean(lose_vals)
    m1_mean = mean(m1_vals)
    # Calibration gap: how far the value head separates won from lost positions.
    # Grows toward +2.0 as the head calibrates; ~0 means it can't tell them apart.
    gap = (
        (win_mean - lose_mean)
        if (win_mean is not None and lose_mean is not None)
        else None
    )

    rec = {
        "time": datetime.now().strftime("%F %T"),
        "mate_in_1_mean": round(m1_mean, 3) if m1_mean is not None else None,
        "winning_mean": round(win_mean, 3) if win_mean is not None else None,
        "losing_mean": round(lose_mean, 3) if lose_mean is not None else None,
        "calib_gap": round(gap, 3) if gap is not None else None,
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

    print(
        f"RESULT_CALIB: mate_in_1_mean={rec['mate_in_1_mean']} "
        f"winning_mean={rec['winning_mean']} losing_mean={rec['losing_mean']} "
        f"gap={rec['calib_gap']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
