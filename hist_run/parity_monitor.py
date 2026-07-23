#!/usr/bin/env python3
"""Parity monitor for the chess history-encoder bootstrap run.

Periodically pits the latest history-encoder trainee (network A, 63-plane
ChessEncoderV2History) against the frozen legacy 1432 champion (network B,
19-plane ChessEncoderV1) via run_arena's new per-network encoder args, logs the
score, and - once the trainee reaches parity with 1432 or plateaus - switches
the training loop from phase A (frozen-generator bootstrap) to phase B (normal
self-improvement) by rewriting hist_run/active_config and restarting the
trainer (the loop picks the new config up on restart).

Each arena is GATED on GPU free-memory headroom and kept to a small footprint,
because a past run OOM-killed training by running an elo check during a
self-play memory peak (see project memory 'arena-concurrent-oom'): arenas run
only at troughs.
"""

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNDIR = ROOT / "hist_run"
ARENA_BIN = ROOT / "build/engine/profiling/run_arena"
HIST_TRT = ROOT / "checkpoints_hist/chess/tensorrt/chess_AZNetwork_hist_0.pt_trt"
HIST_SCRIPTED = (
    ROOT / "checkpoints_hist/chess/scripted/chess_AZNetwork_hist_0.pt_scripted"
)
CHAMP_1432 = (
    ROOT / "checkpoints/chess/old_checkpoint/chess_AZNetwork_20260718_1432.pt_trt"
)
ACTIVE_CONFIG = RUNDIR / "active_config"
PHASE_B_CONFIG = "training_params/chess_params_hist_normal.json"
PHASE_FLAG = RUNDIR / "phase"  # "A" or "B"
PARITY_LOG = RUNDIR / "parity_log.jsonl"
STATUS_LOG = RUNDIR / "train_status.log"

INTERVAL_SEC = 1800  # 30 min between attempts
GPU_FREE_MIN_MB = 10000  # only arena when the GPU is deep in a trough
ARENA_GAMES = 40
ARENA_THREADS = 6
ARENA_SIMS = 128
ARENA_BATCH = 16
# Parity: trainee scores >= this vs 1432 ("equally strong or better", with a
# small noise allowance) on two consecutive arenas.
PARITY_SCORE = 0.47
PARITY_STREAK = 2
# Plateau: many arenas in, no meaningful new high for a long stretch.
PLATEAU_MIN_ARENAS = 14
PLATEAU_WINDOW = 6
PLATEAU_DELTA = 0.03


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def status(msg):
    line = f"[{now()}] [parity-monitor] {msg}\n"
    with open(STATUS_LOG, "a") as f:
        f.write(line)


def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def latest_a_model():
    # Prefer the SCRIPTED engine: it is (re)written every iteration, so it is
    # always the current net, whereas in a frozen-generator bootstrap the TRT
    # engine is compiled rarely or never (tensorrt_every=0) and any TRT on disk
    # may be stale. Scripted is FP32 and a touch slower than TRT but perfectly
    # fine for a 40-game arena, and it never misrepresents the trainee's
    # strength with an out-of-date engine.
    if HIST_SCRIPTED.exists():
        return str(HIST_SCRIPTED)
    if HIST_TRT.exists():
        return str(HIST_TRT)
    return None


def run_arena():
    a_model = latest_a_model()
    if a_model is None or not CHAMP_1432.exists():
        return None
    cmd = [
        str(ARENA_BIN),
        "chess",
        a_model,
        str(CHAMP_1432),
        str(ARENA_GAMES),
        str(ARENA_THREADS),
        "512",
        str(ARENA_SIMS),
        str(ARENA_BATCH),
        "1",
        "16",
        "1000000",
        "4",  # encoder_history_a  -> ChessEncoderV2History(4) for the trainee
        "0",  # encoder_history_b  -> ChessEncoderV1 for the legacy 1432
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        status("arena timed out after 30 min; skipping")
        return None
    text = out.stdout + out.stderr
    m = re.search(r"Score for A:\s*([0-9.]+)%", text)
    if not m:
        status(f"arena produced no parsable score (exit {out.returncode}); skipping")
        return None
    return float(m.group(1)) / 100.0


def switch_to_phase_b(reason):
    ACTIVE_CONFIG.write_text(PHASE_B_CONFIG + "\n")
    PHASE_FLAG.write_text("B\n")
    status(
        f"PARITY/PLATEAU reached ({reason}). Switched active_config -> {PHASE_B_CONFIG}. "
        f"Restarting trainer into phase B (normal self-improvement)."
    )
    # Restart the trainer so it re-reads active_config. The restart loop relaunches it.
    subprocess.run(
        [
            "pkill",
            "-TERM",
            "-f",
            "python -m python --config training_params/chess_params_hist_bootstrap.json",
        ],
        capture_output=True,
    )


def main():
    if not PHASE_FLAG.exists():
        PHASE_FLAG.write_text("A\n")
    status(
        f"parity monitor started (interval {INTERVAL_SEC}s, GPU-free gate "
        f"{GPU_FREE_MIN_MB} MiB, parity>={PARITY_SCORE} x{PARITY_STREAK})."
    )
    scores = []
    streak = 0
    while True:
        time.sleep(INTERVAL_SEC)
        if PHASE_FLAG.read_text().strip() == "B":
            status("phase B active; monitor exiting.")
            return
        free = gpu_free_mb()
        if free < GPU_FREE_MIN_MB:
            status(
                f"GPU free {free} MiB < {GPU_FREE_MIN_MB} MiB (training peak); "
                f"skipping arena this cycle."
            )
            continue
        score = run_arena()
        if score is None:
            continue
        scores.append(score)
        with open(PARITY_LOG, "a") as f:
            f.write(
                json.dumps(
                    {"time": now(), "score_a_vs_1432": score, "arena_idx": len(scores)}
                )
                + "\n"
            )
        elo = -400.0 * (
            __import__("math").log10(1.0 / min(max(score, 1e-3), 1 - 1e-3) - 1.0)
        )
        status(
            f"arena #{len(scores)}: trainee scores {score * 100:.1f}% vs 1432 "
            f"(~{elo:+.0f} Elo)."
        )

        # Parity check
        if score >= PARITY_SCORE:
            streak += 1
        else:
            streak = 0
        if streak >= PARITY_STREAK:
            switch_to_phase_b(
                f"{score * 100:.1f}% >= {PARITY_SCORE * 100:.0f}% x{PARITY_STREAK}"
            )
            return

        # Conservative plateau check
        if len(scores) >= PLATEAU_MIN_ARENAS:
            recent_max = max(scores[-PLATEAU_WINDOW:])
            prior_max = max(scores[:-PLATEAU_WINDOW])
            if recent_max <= prior_max + PLATEAU_DELTA:
                switch_to_phase_b(
                    f"plateau: recent max {recent_max * 100:.1f}% vs prior max "
                    f"{prior_max * 100:.1f}% over {len(scores)} arenas"
                )
                return


if __name__ == "__main__":
    main()
