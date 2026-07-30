#!/usr/bin/env python3
"""Append scalar(s) to the eval TensorBoard run from the cron checks.

Usage:  tb_log.py TAG VALUE [TAG VALUE ...]

Writes to runs/chess_hist_eval with step = int(now) and walltime = now, so the
STEP axis is time-ordered and the WALL/RELATIVE axis lines up with the training
run (runs/chess_hist). Non-numeric values (e.g. "?" from a failed arena matchup)
are silently skipped. Best-effort: any failure exits non-zero without raising,
so a caller using `|| true` never breaks the surrounding check.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGDIR = REPO / "runs" / "chess_hist_eval"


def main(argv) -> int:
    args = argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print("usage: tb_log.py TAG VALUE [TAG VALUE ...]", file=sys.stderr)
        return 2
    pairs = []
    for i in range(0, len(args), 2):
        try:
            pairs.append((args[i], float(args[i + 1])))
        except ValueError:
            continue  # skip non-numeric (e.g. "?" or "PARSE_FAIL")
    if not pairs:
        return 0
    from torch.utils.tensorboard import SummaryWriter

    now = time.time()
    step = int(now)
    writer = SummaryWriter(str(LOGDIR))
    for tag, val in pairs:
        writer.add_scalar(tag, val, step, walltime=now)
    writer.flush()
    writer.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as exc:  # noqa: BLE001 - never break the calling check
        print(f"tb_log: skipped ({exc})", file=sys.stderr)
        sys.exit(1)
