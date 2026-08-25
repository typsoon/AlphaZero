---
name: update-mateusz-arena-anchor
description: Search saved checkpoints for the one that scores best in a head-to-head arena match against mateusz_v2, and update the mateusz_beat_XXXX entry in hist_run/cron/arena_check.sh's OPPONENTS list to point at it if it beats the current entry. Use when the user asks to refresh/update/improve the mateusz_beat arena anchor, or more generally to find the best saved checkpoint against a specific fixed opponent.
---

# Updating the mateusz_beat_XXXX arena anchor

`mateusz_v2` (`mateusz_champions/best_v2/generation-000157.pt_scripted`) has been the
one persistently hard fixed anchor in the arena - the live net has struggled to beat
it consistently (see project memory on the arena's fixed-opponent history). The
`mateusz_beat_XXXX` entry exists as a second, easier-to-beat reference point: a
*saved* checkpoint that has itself beaten `mateusz_v2`, giving a stable "how much
better than a network that once won" signal alongside the harder `mateusz_v2` anchor
itself.

## 1. Confirm the current baseline

```bash
sed -n '/^OPPONENTS=(/,/^)/p' hist_run/cron/arena_check.sh | grep mateusz_beat
```
Note the current path and, if available, its most recent score from
`hist_run/cron/arena_log.jsonl` (`grep '"opponent":"mateusz_beat' ...`).

## 2. Check GPU headroom before running anything

```bash
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits
```
Arena matches share the GPU with live training - only proceed with real headroom
(2500MB+ is the threshold `arena_check.sh` itself uses), and expect each 24-game
match to take longer than usual (~2-4min) since it's not the only GPU consumer.
**Run matches in the background**, not foreground - a 2min foreground timeout will
kill a match that's competing with training for GPU time.

## 3. Build the candidate list

Candidates worth testing:
- The 5 "best" checkpoints kept by [[archive-checkpoints]] in
  `checkpoints/chess/old_checkpoint/` (recent archived snapshots).
- The other fixed anchors already in the arena (`anchor_1053`, `pre_puct`,
  `anchor_0911`, `anchor_0805`) - already-loadable, no extra restore needed.
- Any newly-appeared archive snapshot since the last time this search ran (`find
  checkpoints/chess/old_checkpoint -name "*.pt_scripted" -newermt "<last search
  time>"`).
- Optionally the live net itself, snapshotted the same way `arena_check.sh` does
  (wait for the scripted file's size to settle, then copy) - **but exclude it from
  the final pick**. It isn't a stable checkpoint; using it as a fixed arena opponent
  defeats the purpose (every future reading would just compare the live net to
  itself). Report its score for context only.

## 4. Run the sweep

```bash
MATEUSZ="mateusz_champions/best_v2/generation-000157.pt_scripted"
./build/engine/profiling/run_arena chess "$CANDIDATE_PATH" "$MATEUSZ" 24 6 512 200 16 1 16 1000000 4 4
```
24 games, 6 threads, 200 sims, mcts_batch_size=16 matches `arena_check.sh`'s own
production parameters - keep results comparable to what cron itself reports.
Parse `Score for A: X%` and `Elo difference (A - B): Y` from the output.

## 5. Pick the winner and update

Among **saved** (non-live) candidates only, pick the highest score. If it beats the
current `mateusz_beat_XXXX` entry, edit `hist_run/cron/arena_check.sh`:
- Keep the file already present in `checkpoints/chess/old_checkpoint/` (copy back
  from `/mnt/storage/users/z1201659/old_checkpoints/<dated-folder>/` first if it was
  archived away, and `md5sum` both copies to confirm they match).
- Update the path in the `mateusz_beat_XXXX` line.
- Extend the comment above the entry with what changed and why - date, old vs new
  score/Elo, which candidates were tested - following the existing comment style in
  that file (see the entry's current comment for the pattern: origin story first,
  then the specific replacement rationale appended, not replaced).
- `bash -n hist_run/cron/arena_check.sh` to confirm no syntax break.

## 6. Report

State the old and new scores/Elo, note the margin size relative to typical 24-game
arena noise (roughly ±15-20 points has been observed run-to-run for an identical
matchup - don't oversell a small margin as decisive), and confirm the file change is
live for the next arena cron cycle (no restart needed - `arena_check.sh` re-reads its
own OPPONENTS array fresh every invocation).
