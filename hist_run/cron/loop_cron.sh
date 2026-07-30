#!/bin/bash
# Background scheduler ("cron daemon") for the AlphaZero history-run evaluations.
# Real per-user crontab is not available on this box, so this long-lived loop
# drives the checks instead. Intervals are tracked by elapsed wall-clock time
# (not minute-of-hour modulo) so a 19-minute cadence stays evenly spaced.
#
#   arena_check.sh    every 1 hour   (vs representative champions + last-4h selves)
#   puzzle_check.sh   every 19 min   (puzzle solve-rate + value calibration)
#   memory_check.sh   every 19 min   (training RSS / GPU / system memory)
#   archive_net.sh    every 2 hours  (snapshot the current net into old_checkpoint)
#
# Each task self-gates (GPU headroom + flock) and logs its own RESULT line; this
# loop just triggers them and appends a heartbeat/trigger line to cron.log.
cd "$(dirname "$0")" || exit 1
SCRIPT_DIR="$(pwd)"
CRON_LOG="$SCRIPT_DIR/cron.log"

ARENA_INT=3600     # 1 hour
CHECK_INT=1140     # 19 minutes (puzzle + memory)
ARCHIVE_INT=7200   # 2 hours

log() { echo "[$(date '+%F %T')] $*" >>"$CRON_LOG"; }

trigger() {  # name script
  log "trigger $1"
  ( "$SCRIPT_DIR/$2" >>"$CRON_LOG" 2>&1 ) &
}

log "loop_cron started (pid $$): arena=${ARENA_INT}s checks=${CHECK_INT}s archive=${ARCHIVE_INT}s"

now=$(date +%s)
# Fire the light/gated checks right away for immediate signal; first archive in 2h.
next_arena=$now
next_puzzle=$now
next_memory=$now
next_archive=$((now + ARCHIVE_INT))

while true; do
  now=$(date +%s)

  if [ "$now" -ge "$next_memory" ]; then
    trigger memory  memory_check.sh
    next_memory=$((now + CHECK_INT))
  fi
  if [ "$now" -ge "$next_puzzle" ]; then
    trigger puzzle  puzzle_check.sh
    next_puzzle=$((now + CHECK_INT))
  fi
  if [ "$now" -ge "$next_arena" ]; then
    trigger arena   arena_check.sh
    next_arena=$((now + ARENA_INT))
  fi
  if [ "$now" -ge "$next_archive" ]; then
    trigger archive archive_net.sh
    next_archive=$((now + ARCHIVE_INT))
  fi

  # Sleep to the next minute boundary to avoid drift.
  sec=$(date +%S)
  sleep $(( 60 - 10#$sec ))
done
