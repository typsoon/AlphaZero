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
export LD_LIBRARY_PATH=$(python -c "import torch_tensorrt, os; print(os.path.join(os.path.dirname(torch_tensorrt.__file__), 'lib'))"):$LD_LIBRARY_PATH

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
