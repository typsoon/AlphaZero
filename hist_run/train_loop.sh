#!/bin/bash
# Restart-on-crash wrapper for the chess history-encoder bootstrap run.
#
# Phase A: distills a fresh 63-plane ChessEncoderV2History chess_v2 net from the
# frozen legacy 1432 champion (config chess_params_hist_bootstrap.json,
# resignation off). Phase B: normal self-improvement once the trainee reaches
# parity with 1432 (config chess_params_hist_normal.json - no frozen generator).
#
# The active config is read from hist_run/active_config on every (re)start, so
# the A->B switch is just: echo the phase-B config path into that file (the
# monitor does this automatically at parity/plateau; it can also be done by
# hand). Both configs share checkpoint_dir=checkpoints_hist + stem
# AZNetwork_hist, so the net continues seamlessly across the switch.
cd "$(dirname "$0")/.." || exit 1

# Previously this script relied on the launching shell having already
# activated the venv and set up LD_LIBRARY_PATH/LD_PRELOAD (see
# machine-local skill) - fragile, since a restart from a fresh shell without
# that ambient state would break in confusing ways. Made self-contained here
# instead, mirroring hist_run/cron/common.sh's env setup exactly.
# Machine-local overrides (venv paths differ per machine) - see .env.example.
# Defaults below match this script's original machine, so behavior is
# unchanged unless a .env is present.
REPO="$(pwd)"
set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
VENV_ACTIVATE="${ALPHAZERO_HIST_VENV_ACTIVATE:-/mnt/storage/users/z1201659/.14_ml_venv/bin/activate}"
source "$VENV_ACTIVATE"
VENV_SP="${ALPHAZERO_HIST_VENV_SITE_PACKAGES:-/mnt/storage/users/z1201659/.14_ml_venv/lib/python3.14/site-packages}"
NVIDIA_LIBS=$(printf '%s:' "$VENV_SP"/nvidia/*/lib)
export LD_LIBRARY_PATH="$VENV_SP/torch/lib:${NVIDIA_LIBS}$LD_LIBRARY_PATH"
# torch_tensorrt has no cp314 wheel (see [[venv-py314-cutover]]); only add its
# lib dir to the path when it's actually installed, instead of hard-failing.
TRT_LIB=$(python -c "import torch_tensorrt, os; print(os.path.join(os.path.dirname(torch_tensorrt.__file__), 'lib'))" 2>/dev/null)
[ -n "$TRT_LIB" ] && export LD_LIBRARY_PATH="$TRT_LIB:$LD_LIBRARY_PATH"

# The .14_ml_venv uses the system python3.14 directly (no conda repoint
# needed, unlike .12_ml_venv - see [[venv-py314-breakage-fix]]), so this
# preload isn't strictly required here, but it's a harmless no-op preload of
# the same library the system would resolve anyway - kept for parity/safety.
SYS_LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
[ -e "$SYS_LIBSTDCPP" ] && export LD_PRELOAD="$SYS_LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"

RUNDIR=hist_run
RUN_LOG="$RUNDIR/train_run.log"
STATUS_LOG="$RUNDIR/train_status.log"
CORE_DIR="$RUNDIR/cores"
ACTIVE_CONFIG_FILE="$RUNDIR/active_config"
mkdir -p "$CORE_DIR"

# Same glibc arena cap as attempt #14: with 50 self-play threads the default
# (8 x cores) lets RSS creep via per-thread arena fragmentation. 8 arenas bound
# it without serious lock contention.
export MALLOC_ARENA_MAX=8
ulimit -c unlimited

attempt=0
while true; do
  attempt=$((attempt + 1))
  CONFIG=$(cat "$ACTIVE_CONFIG_FILE" 2>/dev/null)
  if [ -z "$CONFIG" ]; then
    CONFIG="training_params/chess_params_hist_bootstrap.json"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting attempt #$attempt with config $CONFIG" >>"$STATUS_LOG"
  python -m python --config "$CONFIG" >>"$RUN_LOG" 2>&1
  exit_code=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt #$attempt (config $CONFIG) exited with code $exit_code" >>"$STATUS_LOG"
  if [ -f core ]; then
    mv core "$CORE_DIR/core.attempt${attempt}.$(date '+%Y%m%d_%H%M%S')"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saved core dump for attempt #$attempt" >>"$STATUS_LOG"
  fi
  # A clean, deliberate stop (exit 0 from a STOP sentinel) ends the loop;
  # crashes (segfault 139, SIGKILL/OOM 137, SIGTERM 143) auto-restart.
  if [ -f "$RUNDIR/STOP" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STOP sentinel present - exiting loop." >>"$STATUS_LOG"
    break
  fi
  sleep 3
done
