#!/bin/bash
# Arena check: play the current history-encoder net (A, 63-plane, encoder
# history 4) against a representative spread of past champions (B, 19-plane,
# encoder 0). GPU-gated + flock-serialized. Prints a one-line RESULT summary.
source "$(dirname "$0")/common.sh"
LOG="$REPO/hist_run/cron/arena.log"
JSONL="$REPO/hist_run/cron/arena_log.jsonl"

# label:engine:enc_b  -- enc_b is the OPPONENT's encoder history (0 = 19-plane
# ChessEncoderV1 for the legacy champions; 4 = 63-plane history encoder for our
# own recent snapshots). Side A (the live hist net) is always encoder 4.
#
# Fixed strength spread: an early net, the mid-run anchor 0716 (see project
# memory 'arena-sample-size-noise'), and the parity target 1432 -- all 19-plane.
# .pt_trt here is the onnx backend's native raw TensorRT engine (network.py's
# backend="onnx"), not the original torch_tensorrt-compiled one - the latter
# has no cp314 wheel, so its custom TorchScript class
# (torch.classes.tensorrt.Engine) never gets registered by a 3.14-built
# run_arena, throwing "Unknown type name" on load. Recompiled 2026-07-30 from
# each checkpoint's .pt weights via network.py's backend="onnx" path (see
# [[native-tensorrt-engine-loading]]) - loads and runs fine through
# basic_infer.cpp's TensorRTInferenceBackend.
OPPONENTS=(
    "early_0714_0936:$OLD_DIR/chess_AZNetwork_20260714_0936.pt_trt:0"
    "mid_0716:$OLD_DIR/chess_AZNetwork_20260716_0716.pt_trt:0"
    "champ_1432:$OLD_DIR/chess_AZNetwork_20260718_1432.pt_trt:0"
    # The 04:00 net the current run was restarted from (2026-07-27) - a fixed
    # progress anchor: >50% means training has improved over its own start
    # point. Same architecture/encoder as side A (enc 4), .pt_scripted so no
    # double-TRT-context crash.
    "seed_0400:$OLD_DIR/chess_AZNetwork_hist_20260725_040000.pt_scripted:4"
    # A friend's mature v1 net (Engine-Zoo ckpt_1300, converted 2026-07-28) - a
    # fixed external reference. 19-plane v1, so encoder 0; .pt_scripted so no
    # double-TRT-context crash against side A. Baseline: mateusz went 15% vs
    # champ_1432 and ~even (55%) vs the current net at conversion time.
    "mateusz:$REPO/mateusz_champions/best_v1/mateusz_champion.pt_scripted:0"
    # Same friend's chess-v2 net (Engine-Zoo generation-000100, converted
    # 2026-07-30 via python.tools.convert_safetensors_v2 - see
    # [[native-tensorrt-engine-loading]]). Same architecture/encoder as side A
    # (enc 4) unlike best_v1 above, so this is a same-arch comparison, not a
    # cross-arch one. Puzzle-eval scored notably below the current run
    # (60.87% vs ~80-85%) at conversion time; verified not a conversion bug
    # (SE-block shapes/semantics match exactly, value head well-calibrated on
    # clear-cut mate threats) - genuinely a weaker checkpoint on this puzzle
    # set. Native onnx-TRT .pt_trt, loads fine via TensorRTInferenceBackend.
    "mateusz_v2:$REPO/mateusz_champions/best_v2/best_v2.pt_trt:4"
    # The pre-collapse peak the run was rolled back to (2026-07-29): a fixed
    # "did it climb past where we restarted?" anchor. >50% means the current net
    # has surpassed the rollback point. Same arch/encoder as side A (enc 4),
    # .pt_scripted so no double-TRT-context crash.
    "rollback_2316:$OLD_DIR/chess_AZNetwork_hist_20260728_231632.pt_scripted:4"
)

# Self-progression opponents: our own archived history-net snapshots from the
# last 4h (archive_net.sh writes these every 2h). Same architecture/encoder, so
# a >50% score vs a 2-4h-old self means the net is genuinely improving over its
# recent past (a moving anchor, unlike the fixed champions). Encoder 4 both
# sides; both are .pt_scripted so no double-TRT-context crash. Most recent 3,
# newest first, excluding the live checkpoint itself.
while IFS= read -r snap; do
    [ -n "$snap" ] || continue
    base="$(basename "$snap")"
    # chess_AZNetwork_hist_20260720_190802.pt_scripted -> self_1908
    hhmm="$(echo "$base" | grep -oE '_[0-9]{8}_[0-9]{6}\.' | grep -oE '[0-9]{6}\.' | cut -c1-4)"
    OPPONENTS+=("self_${hhmm}:${snap}:4")
done < <(find "$OLD_DIR" -name "chess_AZNetwork_hist_*.pt_scripted" ! -name "*ref*" -mmin -240 \
    -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -3 | cut -d' ' -f2-)
GAMES=24
THREADS=6
SIMS=200

exec 9>"$LOCK"
if ! flock -w 900 9; then
    echo "RESULT: SKIP arena: another GPU task holds the lock (timed out)"
    exit 0
fi
free=$(gpu_free_mb)
if [ "${free:-0}" -lt "$GPU_MIN_MB" ]; then
    msg="SKIP arena: GPU free ${free} MiB < ${GPU_MIN_MB} (training peak)"
    echo "[$(date '+%F %T')] $msg" >>"$LOG"
    echo "RESULT: $msg"
    exit 0
fi
if [ ! -f "$HIST_SCRIPTED" ]; then
    echo "RESULT: SKIP arena: $HIST_SCRIPTED not found"
    exit 0
fi

# The live scripted net is overwritten by the trainer every ~104s; loading it
# mid-write makes a matchup fail silently (empty score). Snapshot it ONCE to a
# stable path (waiting for the size to settle so we never copy a partial file)
# and use that as side A for every matchup - also makes all matchups score a
# CONSISTENT net rather than a slightly newer one each time the loop advances.
SIDE_A="$REPO/hist_run/cron/_arena_sideA.pt_scripted"
trap 'rm -f "$SIDE_A"' EXIT
prev=-1
for _ in 1 2 3 4 5; do
    cur=$(stat -c %s "$HIST_SCRIPTED" 2>/dev/null || echo 0)
    [ "$cur" -gt 0 ] && [ "$cur" = "$prev" ] && break
    prev="$cur"
    sleep 1
done
cp "$HIST_SCRIPTED" "$SIDE_A"

summary=""
tb_pairs=()
for entry in "${OPPONENTS[@]}"; do
    label="${entry%%:*}"
    rest="${entry#*:}"
    opp="${rest%:*}"
    enc_b="${rest##*:}"
    [ -f "$opp" ] || {
        summary="$summary ${label}=MISSING"
        continue
    }
    score=""
    for attempt in 1 2; do
        out=$("$ARENA_BIN" chess "$SIDE_A" "$opp" "$GAMES" "$THREADS" 512 "$SIMS" 16 1 16 1000000 4 "$enc_b" 2>&1)
        score=$(echo "$out" | grep -oE "Score for A: [0-9.]+%" | head -1 | grep -oE "[0-9.]+")
        elo=$(echo "$out" | grep -oE "Elo difference \(A - B\): [+-]?[0-9]+" | grep -oE "[+-]?[0-9]+$" | head -1)
        [ -n "$score" ] && break # retry once on a parse failure (transient crash)
    done
    score="${score:-?}"
    echo "[$(date '+%F %T')] vs ${label}: A=${score}% (Elo ${elo:-?})" >>"$LOG"
    echo "{\"time\":\"$(date '+%F %T')\",\"opponent\":\"$label\",\"score_a_pct\":\"$score\",\"elo\":\"${elo:-}\"}" >>"$JSONL"
    summary="$summary ${label}=${score}%"
    # Collect for TensorBoard as a win fraction (0-1); skip failed matchups.
    if [[ "$score" =~ ^[0-9.]+$ ]]; then
        tb_pairs+=("eval/arena/${label}" "$(awk "BEGIN{print $score/100}")")
    fi
done
echo "RESULT: arena vs ${#OPPONENTS[@]} nets ->$summary"
# Push this round's scores to TensorBoard (runs/chess_hist_eval); best-effort.
[ ${#tb_pairs[@]} -gt 0 ] && python "$REPO/hist_run/cron/tb_log.py" "${tb_pairs[@]}" 2>/dev/null || true
