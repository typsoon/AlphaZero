#!/bin/bash
# Shared config for the history-run scheduled loop tasks (archive / arena /
# puzzle). Sourced by each task script.
REPO=/home/z1201659/AlphaZero/really_newest/AlphaZeroDev
VENV=/mnt/storage/users/z1201659/.12_ml_venv/bin/activate

HIST_PT="$REPO/checkpoints_hist/chess/chess_AZNetwork_hist_0.pt"
HIST_SCRIPTED="$REPO/checkpoints_hist/chess/scripted/chess_AZNetwork_hist_0.pt_scripted"
OLD_DIR="$REPO/checkpoints/chess/old_checkpoint"
ARENA_BIN="$REPO/build/engine/profiling/run_arena"
INFER_BIN="$REPO/build/inference_server/inference_server"

# Only run GPU work when the card is deep enough in a trough - the history run
# is light (~2-3 GB) but a training self-play peak plus a concurrent eval once
# OOM-killed training (project memory 'arena-concurrent-oom').
GPU_MIN_MB=8000
LOCK="$REPO/hist_run/cron/gpu.lock"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}
