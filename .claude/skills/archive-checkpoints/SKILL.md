---
name: archive-checkpoints
description: Choose the 5 best checkpoints in checkpoints/chess/old_checkpoint/, keep everything hist_run/cron/arena_check.sh's OPPONENTS list references, and move the rest to a new dated folder under /mnt/storage/users/z1201659/old_checkpoints/ (next to the venv). Use when the user asks to clean up / archive / free space in the checkpoint directory, or when disk-quota pressure on /home needs relief.
---

# Archiving old checkpoints

This relieves pressure on the per-user NFS quota on `/home` (see
[[checkpoint-corruption-recovery]] for what happens when that quota is exceeded -
silent crash-looping on a corrupted checkpoint) by moving bulk archived checkpoint
data to `/mnt/storage`, which isn't quota-constrained the same way.

## 1. Inventory what's there

```bash
ls -la checkpoints/chess/old_checkpoint/ | sort
du -sh checkpoints/chess/old_checkpoint/
```

## 2. Delete confirmed-corrupt casualties outright

Files with no `.pt_scripted` sibling, or that fail to load, are junk from a
mid-write crash (see [[checkpoint-corruption-recovery]]) - not worth archiving:

```bash
source hist_run/cron/common.sh
for f in checkpoints/chess/old_checkpoint/*.pt; do
  python -c "import torch; torch.load('$f', map_location='cpu', weights_only=True)" \
    2>/dev/null || echo "CORRUPT: $f"
done
```
Delete anything flagged corrupt (`rm`, not archive - it's not recoverable data).

## 3. Identify the arena-referenced keep-list (never move these)

```bash
sed -n '/^OPPONENTS=(/,/^)/p' hist_run/cron/arena_check.sh | grep -oE '\$OLD_DIR/[^:"]+'
```
Every path this prints must stay in `checkpoints/chess/old_checkpoint/` - moving one
breaks the next arena cron run silently (it just prints `SKIP` for that opponent).

## 4. Choose the "5 best" checkpoints

Absent per-checkpoint arena data for most of the 2-hourly `archive_net.sh` snapshots,
the best available proxy for "best" is **most recent** (training has generally been
improving or flat over any given multi-day window - see project memory on arena
trends). Verify each candidate actually loads before trusting it as "best" - the
most recent 1-2 snapshots are exactly the ones most likely to be corrupted if a
quota incident just happened (see step 2's loop, run it over candidates, not just
the whole directory, if step 2 already ran clean):

```bash
ls checkpoints/chess/old_checkpoint/*.pt | sort | tail -8   # newest first once reversed
```
Pick the 5 newest that load cleanly. If the user has a stronger signal available
(e.g. a recent arena sweep across several saved checkpoints), prefer that over pure
recency - see [[update-mateusz-arena-anchor]] for the pattern of directly A/B testing
saved checkpoints against a fixed opponent instead of guessing from recency.

## 5. Move everything else

```bash
DEST=/mnt/storage/users/z1201659/old_checkpoints/$(date +%m-%d_%H-%M)
mkdir -p "$DEST"
cd checkpoints/chess/old_checkpoint
for f in *; do
  # skip anything in the keep-list (5 best + arena-referenced) built above
  mv "$f" "$DEST/"
done
```
This follows the pre-existing convention at `/mnt/storage/users/z1201659/old_checkpoints/`
(dated `MM-DD_HH-MM` subfolders of loose files) - don't invent a new naming scheme.

## 6. Verify before declaring done

```bash
# every arena-referenced file must still resolve:
sed -n '/^OPPONENTS=(/,/^)/p' hist_run/cron/arena_check.sh | grep -oE '\$OLD_DIR/[^:"]+' \
  | sed 's#\$OLD_DIR#checkpoints/chess/old_checkpoint#' | xargs -I{} sh -c '[ -f "{}" ] && echo OK {} || echo MISSING {}'

# quota headroom actually improved:
df -h /home
dd if=/dev/zero of=checkpoints/.quota_test bs=1M count=100 2>&1 | tail -2 && rm -f checkpoints/.quota_test
```

Report final size before/after and confirm no arena opponent went missing.
