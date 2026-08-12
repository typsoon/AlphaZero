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
# early_0714_0936, seed_0400, and rollback_2316 were removed 2026-08-05 09:11
# - all three had saturated at ~98-100% win rate for the live net over the
# preceding ~24h (early_0714_0936: min 95.8%/mean 99.8% across 19 readings;
# seed_0400: min 95.8%/mean 98.6%; rollback_2316: min 91.7%/mean 97.9%),
# i.e. no longer discriminating - every round just re-confirmed what was
# already known. mid_0716 and champ_1432 were KEPT despite occasionally also
# hitting 100%, since they still show real variance down to 70-79% and so
# still carry signal. Replaced with a fresh anchor_0911 snapshot (below) so
# the arena keeps a moving "how far since now" reference instead of shrinking
# to fewer live comparisons.
OPPONENTS=(
    # mid_0716 removed 2026-08-06 10:53, replaced by anchor_1053 below - see
    # that entry's comment for why. champ_1432 itself removed 2026-08-11 -
    # saturated (91-100% for days), no longer discriminating.
    # A fixed snapshot of the live hist net itself from 2026-08-06 10:53,
    # replacing mid_0716 above - taken right at the temperature=1.4 cutover
    # (training_params/chess_params_hist_fresh_puct.toml), after ~1h33m of
    # an unplanned interim run at mcts_simulations=800/use_gumbel_search=false
    # with no temperature yet (see project memory for the full story: a
    # restart landed on this config by accident before the PUCT variant was
    # fully tuned). Same architecture/encoder as side A (enc 4).
    "anchor_1053:$OLD_DIR/chess_AZNetwork_hist_20260806_105337.pt_scripted:4"
    # A fixed snapshot from 2026-08-06 08:00 - the last archive_net.sh
    # snapshot taken BEFORE the accidental switch to PUCT (09:15 the same
    # day), i.e. still purely Gumbel-search-based with the originally
    # intended chess_params_hist_fresh settings. Fixed "how does everything
    # since starting PUCT experiments compare to where the Gumbel run left
    # off" reference. Same architecture/encoder as side A (enc 4).
    "pre_puct:$OLD_DIR/chess_AZNetwork_hist_20260806_080000.pt_scripted:4"
    # A fixed snapshot of the live hist net itself from 2026-08-05 09:11 (the
    # value_loss_weight 0.25->1.0 cutover point - see
    # training_params/chess_params_hist_fresh.json), replacing the three
    # saturated fixed anchors removed above. Same architecture/encoder as
    # side A (enc 4), .pt_scripted so no double-TRT-context crash.
    "anchor_0911:$OLD_DIR/chess_AZNetwork_hist_20260805_091142.pt_scripted:4"
    # A fixed snapshot of the live hist net itself from 2026-08-05 00:00 (the
    # max_replay_reuse_per_iteration=1.0 cutover point), kept alongside
    # anchor_0911 above as a second fixed "since when" reference at a
    # different point in time. Same architecture/encoder as side A (enc 4).
    "anchor_0805:$OLD_DIR/chess_AZNetwork_hist_20260805_000000.pt_scripted:4"
    # Same friend's chess-v2 net, now on generation-000157 (superseded
    # generation-000100/best_v2.pt_trt on 2026-08-04 - a newer, notably
    # stronger checkpoint from the same friend run). Same architecture/encoder
    # as side A (enc 4), so a same-arch comparison, not a cross-arch one.
    # Converted via python.tools.convert_safetensors_v2 (see
    # [[native-tensorrt-engine-loading]]). No .pt_trt yet (TensorRT compile
    # hit GPU OOM from a concurrent job at conversion time) - .pt_scripted
    # works fine and also avoids the double-TRT-context crash. This has been
    # the one persistently negative fixed anchor since the swap (score
    # 12.5-37.5%, Elo -89 to -338 across every reading 2026-08-04 23:00
    # onward) - the live net has not caught up to it despite dominating
    # every other fixed anchor.
    "mateusz_v2:$REPO/mateusz_champions/best_v2/generation-000157.pt_scripted:4"
    # Originally added 2026-08-11 as a fixed snapshot standing in for the
    # live net at the moment it recorded the only >50% score ever against
    # the current mateusz_v2 (gen-000157): 58.3%/+58 Elo at 2026-08-10
    # 01:13:48 - see [[update-mateusz-arena-anchor]] skill for the general
    # procedure this entry now follows. Label kept as mateusz_beat_0810
    # across updates (not re-dated per swap) so its arena_log.jsonl history
    # stays one continuous series. Swap history:
    #   2026-08-11 (1st): 2026-08-10 02:00 archive (nearest of the two
    #     bracketing 2h snapshots to the original 01:13:48 moment, which
    #     wasn't itself archived) - 43.8%/-44 Elo vs mateusz_v2.
    #   2026-08-11 (2nd): 2026-08-10 16:00 archive, found via an A/B search
    #     across the 5 "best" archives + other fixed anchors - 45.8%/-29 Elo
    #     (live net's own 64.6% excluded - not a stable checkpoint).
    #   2026-08-11 (3rd, current): 2026-08-11 10:00 archive - a freshly
    #     appeared snapshot beat the 16:00 one decisively (58.3%/+58 Elo vs
    #     37.5%/-89 in a direct A/B, a 20.8pt/147 Elo gap well outside the
    #     ~15-20pt noise floor of a single 24-game match).
    "mateusz_beat_0810:$OLD_DIR/chess_AZNetwork_hist_20260811_100001.pt_scripted:4"
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
