---
name: checkpoint-corruption-recovery
description: Diagnose and recover the hist chess training loop (hist_run/train_loop.sh) when it is crash-looping on a corrupted checkpoint file. Use when train_status.log shows rapid repeated "Starting attempt #N" / "exited with code 1" lines, or train_run.log shows an EOFError from torch's weights_only unpickler during AlphaZeroNetwork.load_az_network.
---

# Recovering from a corrupted checkpoint crash-loop

## Root cause

`checkpoints_hist/chess/chess_AZNetwork_hist_0.pt` (or the equivalent legacy
`checkpoints/chess/chess_AZNetwork_N.pt`) gets torn mid-write — usually because the
per-user NFS disk quota on `/home` was hit during `network.save_az_network()`. The
result is a 0-byte or truncated `.pt` file. `hist_run/train_loop.sh` immediately
restarts the training process on failure with no backoff, so it crash-loops forever
(every ~3-5s) until someone notices — this has gone undetected for **30+ hours**
(~21,000 failed attempts) before. `df -h /home` can show plenty of headroom overall
while the per-user quota is still hit — don't trust `df` alone to rule out quota
pressure.

## Diagnose

```bash
ps aux | grep "python -m python" | grep -v grep      # process alive? RSS climbing?
tail -30 hist_run/train_status.log                    # rapid Starting/exited cycling?
tail -40 hist_run/train_run.log                       # look for EOFError in
                                                        # torch/_weights_only_unpickler.py
ls -la checkpoints_hist/chess/chess_AZNetwork_hist_0.pt   # 0 bytes or much smaller
                                                            # than neighboring slots?
```

To find when the loop actually started (it's often much earlier than "now"):

```bash
grep -n "2026-08-01" hist_run/train_status.log | grep "exited with code" | head -3
```

## Recover

`python/checkpoint_manager.py`'s `add_checkpoint()` rotates slot N -> N+1 *before*
writing new weights to slot 0, so slot 1 is always the previous-good checkpoint.
Restore from it:

```bash
source hist_run/cron/common.sh   # activates the venv so `python`/torch resolve

# 1. Verify the donor slot is itself intact BEFORE trusting it:
python -c "
import torch
ckpt = torch.load('checkpoints_hist/chess/chess_AZNetwork_hist_1.pt', map_location='cpu', weights_only=True)
print('OK, keys:', list(ckpt.keys())[:5])
"

# 2. Restore slot 0 from slot 1 (both the raw weights and the scripted fallback):
cp checkpoints_hist/chess/chess_AZNetwork_hist_1.pt \
   checkpoints_hist/chess/chess_AZNetwork_hist_0.pt
cp checkpoints_hist/chess/scripted/chess_AZNetwork_hist_1.pt_scripted \
   checkpoints_hist/chess/scripted/chess_AZNetwork_hist_0.pt_scripted
```

Don't skip the `.pt_trt` (TensorRT) slot restore — it's fine to leave missing; it
rebuilds automatically on the next successful `add_checkpoint()` call as long as
`tensorrt_every` in the training config is nonzero.

**If `cp` reports "Disk quota exceeded" on close**, don't assume the copy failed or
is corrupt — the quota can be a transient/grace-period hiccup. Verify before
concluding anything:

```bash
md5sum checkpoints_hist/chess/chess_AZNetwork_hist_1.pt \
       checkpoints_hist/chess/chess_AZNetwork_hist_0.pt   # must match
python -c "import torch; torch.load('checkpoints_hist/chess/chess_AZNetwork_hist_0.pt', map_location='cpu', weights_only=True); print('load OK')"
```
A matching md5 + successful load means the copy landed fine despite the warning.

## Confirm recovery

Watch for a `Starting attempt #N` line in `train_status.log` with no immediate
`exited` line following it, then confirm real progress:

```bash
tail -f hist_run/train_run.log   # expect "Games played: N/500" climbing
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```
RSS climbing steadily in `ps aux` (not stuck at a few hundred MB) plus GPU
utilization near 100% both confirm the process is actually training, not just
alive.

## Don't stop here

This only fixes the symptom. The recurring root cause is `/home` disk-quota
pressure — if this keeps recurring, look at what's regrowing on `/home` (rotating
checkpoints archive dirs, build trees) vs what's genuinely static and should live on
`/mnt/storage` instead. Also flag to the user that the crash-loop has no
alerting — a corrupted checkpoint can silently eat a full day of training time
before anyone checks on it.
