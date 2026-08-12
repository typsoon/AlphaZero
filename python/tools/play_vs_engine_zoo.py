"""Plays a live AlphaZeroDev `inference_server` against engine-zoo's UCI
engine (`engine-zoo-uci`) for a real head-to-head game.

Both sides are genuine UCI engines, driven symmetrically with python-chess's
own `chess.engine` module - no custom protocol code lives in this script at
all:

  - engine-zoo's side: `run_engine_zoo_uci.sh`, a small wrapper around the
    `engine-zoo-uci` binary (see that script's own comments) that sets up
    the LD_LIBRARY_PATH/LD_PRELOAD its CUDA support needs at runtime -
    without it, engine-zoo-uci silently falls back to reporting "CUDA is
    not available" even on a machine with a working GPU, because it links
    libtorch_cpu.so/libc10.so directly rather than the umbrella libtorch.so
    Python loads, and the linker's --as-needed default drops the
    -ltorch_cuda flag torch-sys's build.rs emits.
  - AlphaZeroDev's side: `gameplay_server/dist/tui/az-uci.js`, a small
    TypeScript UCI wrapper (see that file's own docstring) around
    inference_server - reusing gameplay_server's actual production code
    (ChessBoard + AlphaZeroAgent + pickBestLegalAction, the same pieces
    playAITurn uses) rather than reimplementing the request/response
    handling here. inference_server itself never picks a move - /predict
    only returns a policy distribution + value - so *something* has to do
    that evaluate()+argmax step; az-uci.js is that something, now speaking
    real UCI instead of gameplay_server's HTTP/WebSocket API.

Usage:

  # 1. Start your own engine (from the repo root):
  doit run_inference_server --network_path <checkpoint> \\
      --params_file inference_params/inference_params.toml

  # 2. Build the UCI wrapper for AlphaZeroDev's side (once):
  cd gameplay_server && npm run build

  # 3. Build engine-zoo-uci once, against a venv with torch==2.11.0+cu130
  #    (tch 0.24's pinned version - see engine-zoo's README):
  cd /path/to/engine-zoo
  source /path/to/that/venv/bin/activate
  LIBTORCH_USE_PYTORCH=1 cargo build --release --bin engine-zoo-uci

  # 4. Play them against each other:
  python -m python.tools.play_vs_engine_zoo \\
      --az-socket /tmp/alphazero-inference/chess/<net>/<hash>.sock \\
      --ez-wrapper /path/to/engine-zoo/run_engine_zoo_uci.sh \\
      --ez-python /path/to/that/venv/bin/python3 \\
      --ez-run-dir /path/to/engine-zoo/runs/chess-puct-wdl-tensorrt-v5 \\
      --ez-checkpoint /path/to/engine-zoo/runs/chess-puct-wdl-tensorrt-v5/checkpoints/generation-000100.safetensors \\
      --az-plays white --pgn-out game.pgn
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parents[2]
AZ_UCI_JS = REPO_ROOT / "gameplay_server" / "dist" / "tui" / "az-uci.js"


def start_az_engine(socket_path: str) -> chess.engine.SimpleEngine:
    if not AZ_UCI_JS.is_file():
        raise SystemExit(
            f"{AZ_UCI_JS} not found - run 'npm run build' in gameplay_server/ first."
        )
    return chess.engine.SimpleEngine.popen_uci(
        ["node", str(AZ_UCI_JS), "--socket", socket_path]
    )


def start_ez_engine(
    wrapper: str,
    run_dir: str,
    checkpoint: str,
    simulations: int,
    device: str,
    python: str | None,
) -> chess.engine.SimpleEngine:
    # run_engine_zoo_uci.sh resolves torch's install location by running
    # `python3 -c "import torch"` - which, inherited from *this* process's
    # environment, would be whatever venv this orchestrator itself is
    # running under (python-chess's), not necessarily the one with
    # torch==2.11.0+cu130. ENGINE_ZOO_PYTHON overrides that without
    # requiring this process to re-activate anything.
    env = dict(os.environ)
    if python:
        env["ENGINE_ZOO_PYTHON"] = python
    return chess.engine.SimpleEngine.popen_uci(
        [
            wrapper,
            "--run-dir",
            run_dir,
            "--checkpoint",
            checkpoint,
            "--simulations",
            str(simulations),
            "--device",
            device,
        ],
        env=env,
    )


def play_game(
    az_engine: chess.engine.SimpleEngine,
    ez_engine: chess.engine.SimpleEngine,
    az_plays_white: bool,
    max_moves: int,
) -> chess.Board:
    board = chess.Board()
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        az_turn = (board.turn == chess.WHITE) == az_plays_white
        mover = "AlphaZeroDev" if az_turn else "engine-zoo"
        engine = az_engine if az_turn else ez_engine

        t0 = time.monotonic()
        result = engine.play(board, chess.engine.Limit())
        elapsed = time.monotonic() - t0

        move = result.move
        if move is None or move not in board.legal_moves:
            print(
                f"{mover} proposed an illegal/null move '{move}' in position "
                f"{board.fen()} - aborting.",
                file=sys.stderr,
            )
            break

        san = board.san(move)
        board.push(move)

        move_no = ply // 2 + 1
        side = "." if board.turn == chess.BLACK else "..."
        print(f"{move_no}{side} {san:<8} [{mover}, {elapsed:.1f}s]")
        ply += 1

    return board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--az-socket",
        required=True,
        help="Path to a running AlphaZeroDev inference_server's .sock file",
    )
    parser.add_argument(
        "--ez-wrapper",
        required=True,
        help="Path to engine-zoo's run_engine_zoo_uci.sh (not the raw engine-zoo-uci binary - "
        "the wrapper sets up the LD_PRELOAD its CUDA support needs)",
    )
    parser.add_argument(
        "--ez-python",
        default=None,
        help="Path to a python3 with torch==2.11.0+cu130 installed, if it's not already the "
        "one on this process's own PATH (sets ENGINE_ZOO_PYTHON for the wrapper)",
    )
    parser.add_argument(
        "--ez-run-dir",
        required=True,
        help="engine-zoo experiment run dir (has config/state.json)",
    )
    parser.add_argument(
        "--ez-checkpoint",
        required=True,
        help="Path to the .safetensors checkpoint to serve",
    )
    parser.add_argument(
        "--ez-simulations",
        type=int,
        default=800,
        help="engine-zoo-uci's --simulations (used for every 'go' since we send it with no "
        "nodes/movetime override)",
    )
    parser.add_argument("--ez-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--az-plays",
        choices=["white", "black"],
        default="white",
        help="Which side AlphaZeroDev's engine plays",
    )
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument(
        "--pgn-out", default=None, help="Optional path to save a PGN of the game"
    )
    args = parser.parse_args()

    az_engine = start_az_engine(args.az_socket)
    ez_engine = start_ez_engine(
        args.ez_wrapper,
        args.ez_run_dir,
        args.ez_checkpoint,
        args.ez_simulations,
        args.ez_device,
        args.ez_python,
    )

    az_plays_white = args.az_plays == "white"
    print(
        f"AlphaZeroDev plays {'White' if az_plays_white else 'Black'}; "
        f"engine-zoo-uci plays {'Black' if az_plays_white else 'White'}\n"
    )

    try:
        board = play_game(az_engine, ez_engine, az_plays_white, args.max_moves)
    finally:
        az_engine.quit()
        ez_engine.quit()

    print()
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        print(f"Result: {outcome.result()} ({outcome.termination.name})")
    else:
        print(f"Stopped after {args.max_moves} moves without a decided result.")

    if args.pgn_out:
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = (
            "AlphaZeroDev" if az_plays_white else "engine-zoo (mateusz_v2)"
        )
        game.headers["Black"] = (
            "engine-zoo (mateusz_v2)" if az_plays_white else "AlphaZeroDev"
        )
        with open(args.pgn_out, "w") as f:
            print(game, file=f)
        print(f"PGN written to {args.pgn_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
