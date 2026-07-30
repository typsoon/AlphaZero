#!/bin/bash
# Memory check: snapshot the training run's resident memory plus GPU and system
# memory. No GPU compute, no lock needed - just reads /proc, ps and nvidia-smi,
# so it is safe to run alongside training/arena/puzzle. Prints a one-line RESULT
# for the calling loop to report. The history run has had slow RSS creep and one
# OOM-kill (project memory 'arena-concurrent-oom'), so training RSS is the metric
# to watch here.
source "$(dirname "$0")/common.sh"
LOG="$REPO/hist_run/cron/memory.log"
JSONL="$REPO/hist_run/cron/memory_log.jsonl"

# Training process RSS (the self-play/train master); sum over matching procs.
train_rss_mb=$(ps -o rss= -C python 2>/dev/null \
  | awk '{s+=$1} END {printf "%d", s/1024}')
# More precise: just the `python -m python` master + children.
pids=$(pgrep -f "python -m python" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
if [ -n "$pids" ]; then
  train_rss_mb=$(ps -o rss= -p "$pids" 2>/dev/null | awk '{s+=$1} END {printf "%d", s/1024}')
  train_up=1
else
  train_rss_mb=0
  train_up=0
fi

# GPU memory (used/total, MiB).
read -r gpu_used gpu_total < <(nvidia-smi --query-gpu=memory.used,memory.total \
  --format=csv,noheader,nounits 2>/dev/null | head -1 | tr ',' ' ')
gpu_used=${gpu_used:-0}; gpu_total=${gpu_total:-0}

# System memory (MiB): total / available.
sys_total_mb=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null)
sys_avail_mb=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null)

msg="train_rss=${train_rss_mb}MB (up=${train_up}) gpu=${gpu_used}/${gpu_total}MiB sys_avail=${sys_avail_mb}/${sys_total_mb}MB"
echo "[$(date '+%F %T')] $msg" >>"$LOG"
echo "{\"time\":\"$(date '+%F %T')\",\"train_rss_mb\":${train_rss_mb},\"train_up\":${train_up},\"gpu_used_mib\":${gpu_used},\"gpu_total_mib\":${gpu_total},\"sys_avail_mb\":${sys_avail_mb},\"sys_total_mb\":${sys_total_mb}}" >>"$JSONL"
echo "RESULT: memory -> $msg"
