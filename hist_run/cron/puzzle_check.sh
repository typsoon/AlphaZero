#!/bin/bash
# Puzzle check: run the puzzle evaluator on the current history-encoder net
# (63-plane, ChessEncoderV2History) through the inference server. GPU-gated +
# flock-serialized so it never piles onto training or a concurrent arena.
# Prints a one-line RESULT for the calling loop to report/interpret.
source "$(dirname "$0")/common.sh"
LOG="$REPO/hist_run/cron/puzzle.log"
JSONL="$REPO/hist_run/cron/puzzle_log.jsonl"

exec 9>"$LOCK"
if ! flock -w 900 9; then
  echo "RESULT: SKIP puzzle: another GPU task holds the lock (timed out)"
  exit 0
fi
free=$(gpu_free_mb)
if [ "${free:-0}" -lt "$GPU_MIN_MB" ]; then
  msg="SKIP puzzle: GPU free ${free} MiB < ${GPU_MIN_MB} (training peak)"
  echo "[$(date '+%F %T')] $msg" >>"$LOG"
  echo "RESULT: $msg"
  exit 0
fi
if [ ! -f "$HIST_SCRIPTED" ]; then
  echo "RESULT: SKIP puzzle: $HIST_SCRIPTED not found"
  exit 0
fi

cd "$REPO" || exit 1
# shellcheck disable=SC1090
out=$(python -m performance_evaluation.evaluator \
  --network-path "$HIST_SCRIPTED" --inference-binary "$INFER_BIN" \
  --game chess --chess-encoder-history 4 --mcts-search-depth 256 2>&1)
score=$(echo "$out" | grep -oE "Tests Passed: [0-9]+/[0-9]+ \([0-9.]+%\)" | head -1)
if [ -z "$score" ]; then
  score="PARSE_FAIL (see puzzle.log)"
  echo "----- $(date '+%F %T') raw output -----" >>"$LOG"
  echo "$out" | tail -20 >>"$LOG"
fi
echo "[$(date '+%F %T')] $score" >>"$LOG"
echo "{\"time\":\"$(date '+%F %T')\",\"result\":\"$score\"}" >>"$JSONL"
echo "RESULT: puzzles -> $score"
# Push the pass rate (as a 0-1 fraction) to TensorBoard; best-effort.
pct=$(echo "$score" | grep -oE "[0-9.]+%" | head -1 | tr -d '%')
[ -n "$pct" ] && python "$REPO/hist_run/cron/tb_log.py" \
    eval/puzzle_pass_rate "$(awk "BEGIN{print $pct/100}")" 2>/dev/null || true
# Value-head calibration from the just-written results.json (same eval run).
python "$REPO/hist_run/cron/value_calib.py"
