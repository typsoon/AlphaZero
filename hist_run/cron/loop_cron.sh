#!/bin/bash

# Background loop acting as a cron daemon for AlphaZero evaluations
echo "Starting background cron loop..."

while true; do
  min=$(date +%M)
  
  # Puzzle check every 10 minutes
  if [ $((10#$min % 10)) -eq 0 ]; then
    echo "[$(date)] Triggering puzzle_check.sh"
    ./puzzle_check.sh &
  fi
  
  # Arena check at the top of the hour
  if [ "$min" == "00" ]; then
    echo "[$(date)] Triggering arena_check.sh"
    ./arena_check.sh &
  fi
  
  # Sleep until the exact start of the next minute to avoid drifting
  sec=$(date +%S)
  sleep_time=$(( 60 - 10#$sec ))
  sleep $sleep_time
done
