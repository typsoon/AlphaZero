"""Persist the C++ ReplayBuffer across runs, with a compatibility guard.

The buffer is otherwise pure in-memory C++: every process launch builds an empty
one and refills it from self-play (the ~15-20 min ramp visible on every restart).
Saving it lets a restart of the SAME config resume instantly, and lets a
deliberately-changed run reuse old data WHEN the encoding is unchanged.

The hard part is safety: the buffer stores encoded state tensors + sparse MCTS
policy targets. Reloading data whose encoder/action-space differs from the
current net would feed it wrong-shaped inputs or mis-indexed policy targets -
a crash at best, silently corrupt training at worst. So every save writes a
sidecar `<path>.meta.json` recording the encoding signature (`tag`), the action
space size, and the state shape, and load REFUSES on any mismatch rather than
trusting the file. `tag` is built from everything that determines the trainee's
encoding (game + network arch + chess history length), so an encoder swap - the
one change that must never silently reuse a buffer - always trips the guard.

Writes are atomic (temp file + os.replace) so a crash mid-save can never corrupt
the previously-saved good buffer.
"""

import json
import logging
import os
from typing import Optional, Sequence

# Bump if the on-disk tensor layout in ReplayBuffer::save changes.
FORMAT_VERSION = 1


def compute_tag(game: str, network_arch: str, chess_encoder_history: Optional[int]) -> str:
    """Signature of the trainee's input encoding - the thing that must match for
    a saved buffer to be reusable. Trajectories are recorded with the trainee's
    encoder, which is fully determined by (game, network arch, history length),
    so these three pin the encoding exactly. The self-play *generator* encoder is
    deliberately NOT included: it only affects which engine generates games, not
    how the stored states are encoded."""
    return f"{game}|arch={network_arch}|chess_history={chess_encoder_history}"


def _meta_path(path: str) -> str:
    return path + ".meta.json"


def save_buffer(
    buffer,
    path: str,
    tag: str,
    action_size: int,
    state_shape: Optional[Sequence[int]],
) -> None:
    """Atomically write the buffer tensors to `path` and the guard metadata to
    `<path>.meta.json`. The buffer file is written to a temp path and renamed so
    an interrupted save never clobbers the last good file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    buffer.save(tmp)  # C++ torch::save of the packed tensors (GIL released)
    os.replace(tmp, path)

    meta = {
        "format_version": FORMAT_VERSION,
        "tag": tag,
        "action_size": int(action_size),
        "state_shape": list(state_shape) if state_shape is not None else None,
        "size": int(buffer.get_size()),
    }
    meta_tmp = _meta_path(path) + ".tmp"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, _meta_path(path))
    logging.info(
        f"Saved replay buffer ({meta['size']} transitions, tag '{tag}') to '{path}'."
    )


def load_buffer_if_compatible(
    buffer,
    path: Optional[str],
    tag: str,
    action_size: int,
    state_shape: Optional[Sequence[int]],
) -> Optional[int]:
    """Load `path` into `buffer` iff its sidecar metadata is compatible with the
    current run. Returns the loaded transition count, or None if there was
    nothing to load or it was refused (a warning is logged explaining why - a
    refusal is safe, the run just starts from an empty buffer as before)."""
    if not path or not os.path.exists(path):
        logging.info(
            f"No replay buffer to preload (path {'unset' if not path else f'{path!r} absent'});"
            f" starting from empty and filling via self-play."
        )
        return None

    mp = _meta_path(path)
    if not os.path.exists(mp):
        logging.warning(
            f"Replay buffer '{path}' has no '{mp}' sidecar - cannot verify it "
            f"matches this run's encoding, so REFUSING to load it (starting empty)."
        )
        return None

    with open(mp) as f:
        meta = json.load(f)

    problems = []
    if meta.get("format_version") != FORMAT_VERSION:
        problems.append(
            f"format_version {meta.get('format_version')} != {FORMAT_VERSION}"
        )
    if meta.get("tag") != tag:
        problems.append(f"encoding tag {meta.get('tag')!r} != {tag!r}")
    if meta.get("action_size") != int(action_size):
        problems.append(f"action_size {meta.get('action_size')} != {action_size}")
    saved_shape = meta.get("state_shape")
    if (
        state_shape is not None
        and saved_shape is not None
        and list(saved_shape) != list(state_shape)
    ):
        problems.append(f"state_shape {saved_shape} != {list(state_shape)}")

    if problems:
        logging.warning(
            f"REFUSING to load replay buffer '{path}': incompatible with this run "
            f"({'; '.join(problems)}). Starting from an empty buffer. Point "
            f"--replay-buffer-path at a buffer saved by a matching-encoding run, "
            f"or delete the stale file to silence this."
        )
        return None

    buffer.load(path)
    loaded = buffer.get_size()
    logging.info(
        f"Preloaded {loaded} transitions from replay buffer '{path}' "
        f"(tag '{tag}') - skipping the self-play refill ramp."
    )
    return loaded
