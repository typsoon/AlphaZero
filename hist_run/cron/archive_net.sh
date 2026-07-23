#!/bin/bash
# Archive the current history-encoder network into the old-checkpoints dir.
# No GPU work. Prints a one-line RESULT for the calling loop to report.
source "$(dirname "$0")/common.sh"
LOG="$REPO/hist_run/cron/archive.log"
ts=$(date '+%Y%m%d_%H%M%S')

if [ ! -f "$HIST_PT" ]; then
    msg="SKIP archive: $HIST_PT not found (training not running?)"
    echo "[$(date '+%F %T')] $msg" >>"$LOG"
    echo "RESULT: $msg"
    exit 0
fi

dest_pt="$OLD_DIR/chess_AZNetwork_hist_$ts.pt"
cp "$HIST_PT" "$dest_pt"
if [ -f "$HIST_SCRIPTED" ]; then
    cp "$HIST_SCRIPTED" "$OLD_DIR/chess_AZNetwork_hist_$ts.pt_scripted"
fi
msg="archived hist net -> chess_AZNetwork_hist_$ts.pt (+scripted)"
echo "[$(date '+%F %T')] $msg" >>"$LOG"
echo "RESULT: $msg"
