#!/bin/bash
# Shared config for the history-run scheduled loop tasks (archive / arena /
# puzzle). Sourced by each task script.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Machine-local overrides (venv paths differ per machine) - see .env.example.
# Defaults below match this script's original machine, so behavior is
# unchanged unless a .env is present.
set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
VENV_ACTIVATE="${ALPHAZERO_HIST_VENV_ACTIVATE:-/mnt/storage/users/z1201659/.14_ml_venv/bin/activate}"
source "$VENV_ACTIVATE"
# torch 2.13.0+cu126 gets its CUDA runtime from pip nvidia-* packages (not
# torch/lib), and the 2026-07-26 OS upgrade removed /opt/cuda (system now has
# CUDA 13, wrong major for our cu12 binaries). So the C++ tools (run_arena,
# inference_server) need torch/lib + every nvidia/*/lib on the path to resolve
# libtorch* and libcudart.so.12. torch_tensorrt has no cp314 wheel; TensorRT
# itself resolves via RPATH baked into the binaries (see
# [[native-tensorrt-engine-loading]]), no LD_LIBRARY_PATH entry needed.
VENV_SP="${ALPHAZERO_HIST_VENV_SITE_PACKAGES:-/mnt/storage/users/z1201659/.14_ml_venv/lib/python3.14/site-packages}"
NVIDIA_LIBS=$(printf '%s:' "$VENV_SP"/nvidia/*/lib)
export LD_LIBRARY_PATH="$VENV_SP/torch/lib:${NVIDIA_LIBS}$LD_LIBRARY_PATH"

# .14_ml_venv uses the system python3.14 directly (no conda repoint needed,
# unlike .12_ml_venv), so this preload isn't strictly required, but it's a
# harmless no-op preload of the same library the system would resolve anyway
# - kept for parity/safety.
SYS_LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
[ -e "$SYS_LIBSTDCPP" ] && export LD_PRELOAD="$SYS_LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"

HIST_PT="$REPO/checkpoints_hist/chess/chess_AZNetwork_hist_0.pt"
HIST_SCRIPTED="$REPO/checkpoints_hist/chess/scripted/chess_AZNetwork_hist_0.pt_scripted"
OLD_DIR="$REPO/checkpoints/chess/old_checkpoint"
ARENA_BIN="$REPO/build/engine/profiling/run_arena"
INFER_BIN="$REPO/build/inference_server/inference_server"

# Only run GPU work when the card is deep enough in a trough - the history run
# is light (~2-3 GB) but a training self-play peak plus a concurrent eval once
# OOM-killed training (project memory 'arena-concurrent-oom'). Lowered from
# 8000 on 2026-08-03: a live-measured arena run only added ~1-1.3GB on top of
# baseline and peaked at 14.3GB/16.4GB total without incident - the old
# threshold was sized for a single-tenant GPU, but most of the current
# headroom pressure comes from another user's concurrent job (fluctuating
# ~5-11GB), not from our own arena/puzzle footprint.
GPU_MIN_MB=2500
LOCK="$REPO/hist_run/cron/gpu.lock"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}
