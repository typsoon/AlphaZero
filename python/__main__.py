import torch
from .pybind.engine_bind import Connect4, Chess  # pyright: ignore
from .pybind.self_play_bind import self_play_connect4, self_play  # pyright: ignore
from .checkpoint_manager import CheckpointManager
from .injectors import (
    get_chess_encoder,
    get_network,
    get_trainer,
)
from .network import AlphaZeroNetwork
from .train import self_play_and_train_loop
from .replay_persistence import compute_tag
import argparse
import json
import os
import shutil
import logging
from functools import partial
from datetime import datetime
from pathlib import Path
from python.utils import PROJ_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# thread_count = 4
# games_in_each_iteration = 500
# Sized to retain ~4 iterations of games_in_each_iteration=1000 chess games
# (run_training.sh's setting), at an estimated ~80 plies/game - we don't have an
# exact transitions-per-iteration figure (get_size() saturates at capacity, so
# it can't tell us how many transitions self-play actually produced past that
# point). With the sparse ReplayBuffer (~5KB/transition instead of ~85KB dense),
# this is only ~1.5GB, well inside the ~6-10GB of headroom measured during
# self-play's peak memory usage - if the real average game length turns out
# longer than 80 plies, there's room to raise this further.
replay_buffer_size = 1000 * 80 * 4


def prune_old_runs(runs_root: Path, max_runs: int) -> None:
    """Deletes the oldest auto-named run directories under runs_root, keeping at
    most max_runs. Run directory names are timestamps (%Y%m%d-%H%M%S), so
    lexicographic sort is also chronological order. Unlike checkpoints (bounded
    by CheckpointManager), nothing else caps how many of these accumulate across
    repeated invocations - left unchecked they grow forever."""
    if max_runs <= 0 or not runs_root.is_dir():
        return
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    for old_dir in run_dirs[:-max_runs] if len(run_dirs) > max_runs else []:
        shutil.rmtree(old_dir, ignore_errors=True)


def get_args():
    parser = argparse.ArgumentParser(description="My arg parser")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON file with training arguments (keys are argument "
        "names, e.g. 'games_in_each_iteration'). Values in the file become "
        "the new defaults for the arguments below; explicit CLI flags still "
        "take precedence over them.",
    )
    parser.add_argument(
        "--initial-network",
        type=str,
        default=None,
        help="Path to an existing network to initialize from if no checkpoints exist",
    )
    parser.add_argument(
        "--checkpoint_stem",
        type=str,
        default="AZNetwork",
        help="Checkpoint file prefix",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="connect4",
        choices=["connect4", "chess"],
        help="Which game to train",
    )

    parser.add_argument(
        "--games-in-each-iteration",
        type=int,
        default=400,
        help="Number of games in each iteration",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=2000,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--loop-iterations", type=int, default=100, help="Number of loop iterations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=4096, help="Training minibatch size"
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=1e-3,
        help="Adam's initial learning rate (the LR at outer iteration 0). The LR "
        "decays exponentially from here to --lr-lower-bound over the schedule's "
        "decay window, then holds. Set equal to --lr-lower-bound for a flat LR "
        "(recommended for the ongoing run, since restarts reset the schedule to "
        "iteration 0 - see injectors.py)",
    )
    parser.add_argument(
        "--lr-lower-bound",
        type=float,
        default=1e-4,
        help="Floor the learning rate decays to and holds at. Must be <= --initial-lr",
    )

    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=10,
        help="Number of residual blocks when creating a FRESH network. Ignored "
        "when --initial-network loads an existing checkpoint - a checkpoint "
        "carries its own architecture (init_args) and always keeps it, so "
        "restarts of an ongoing run are unaffected by this setting",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=64,
        help="Convolution filter count when creating a FRESH network. Same "
        "semantics as --residual-blocks: checkpoints keep their own "
        "architecture, this only shapes newly created networks",
    )
    parser.add_argument(
        "--network-arch",
        type=str,
        default="legacy",
        choices=["legacy", "chess_v2"],
        help="Architecture for a FRESH network (checkpoints keep their own; "
        "loading dispatches on the checkpoint's architecture tag). 'legacy' is "
        "the original scalar-value ResNet sized by --residual-blocks/"
        "--num-filters; 'chess_v2' is the fixed 12x128 SE-block network with a "
        "spatial policy and WDL value head (chess only)",
    )
    parser.add_argument(
        "--chess-encoder-history",
        type=int,
        default=None,
        choices=[1, 4, 8],
        help="chess_v2 only: use ChessEncoderV2History(history=N) - the "
        "engine-zoo reference's history-stacked input (14*N+7 planes: own/opp "
        "piece planes + 2 repetition-flag planes per retained frame, plus 7 "
        "auxiliary planes for the current position) - instead of the default "
        "single-frame ChessEncoderV1 (19 planes). Sizes a FRESH network's "
        "input_channels/stm_plane_index to match (checkpoints keep their own, "
        "same as --network-arch) and is threaded to self-play so training data "
        "matches inference input. N=4 matches the reference's own default",
    )
    parser.add_argument(
        "--self-play-network",
        type=str,
        default=None,
        help="Frozen-generator bootstrap: path to a FIXED inference engine "
        "(.pt_trt) that generates ALL self-play games while the training net "
        "only learns from them (distillation from a strong net's games, e.g. "
        "to warm up a fresh architecture). The trainee cannot surpass the "
        "frozen generator by much - unset this once it reaches arena parity "
        "to return to normal self-improvement. Default: self-play uses the "
        "latest checkpoint as usual",
    )
    parser.add_argument(
        "--self-play-value-network",
        type=str,
        default=None,
        help="Use a different network for value predictions than for policy in self-play.",
    )
    parser.add_argument(
        "--self-play-value-encoder-history",
        type=int,
        default=None,
        choices=[1, 4, 8],
        help="History length for the value network (default: None for legacy 19-plane encoder).",
    )
    parser.add_argument(
        "--self-play-encoder-history",
        type=int,
        default=None,
        choices=[1, 4, 8],
        help="Frozen-generator bootstrap only: the input encoding the "
        "--self-play-network (generator) expects, when it DIFFERS from the "
        "trainee's --chess-encoder-history. None (default) => the generator "
        "uses ChessEncoderV1 (19 planes) - the correct value for a legacy or "
        "v1-encoder champion (e.g. the 1432 legacy net) generating games for a "
        "fresh history-encoder trainee. Ignored without --self-play-network. "
        "The trajectory recorded for training always uses --chess-encoder-history.",
    )

    parser.add_argument(
        "--thread-count", type=int, default=os.cpu_count(), help="Thread count"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to store historical checkpoints",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="Maximum number of historical checkpoints to keep",
    )
    parser.add_argument(
        "--tensorrt-every",
        type=int,
        default=1,
        help="Compile a TensorRT inference engine for the trainee only every "
        "N checkpoints (0 = never; 1 = every iteration, the default). The .pt "
        "and .pt_scripted engines are always written. Set 0 for a "
        "frozen-generator bootstrap (--self-play-network), where self-play runs "
        "off the frozen generator and the trainee's own TRT engine is unused "
        "until it takes over generation - recompiling it every iteration is "
        "~tens of seconds of pure waste (~27%% of iteration wall-clock at the "
        "chess-v2 sizes). Arena/eval can use the always-current scripted engine.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=512,
        help="Maximum number of moves before a game is truncated",
    )
    parser.add_argument(
        "--mcts-batch-size",
        type=int,
        default=64,
        help="MCTS batch size for neural network inference",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move",
    )
    parser.add_argument(
        "--fast-mcts-simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations for playout-cap-randomization 'fast' "
        "moves (see --full-search-probability) - these moves advance the game "
        "but aren't recorded as training examples",
    )
    parser.add_argument(
        "--full-search-probability",
        type=float,
        default=0.25,
        help="Probability that a move uses the full --mcts-simulations search "
        "(and is recorded as a training example) instead of the cheaper "
        "--fast-mcts-simulations search (playout cap randomization)",
    )
    parser.add_argument(
        "--transposition-cache-entries",
        type=int,
        default=1_000_000,
        help="Slot count of the self-play NN-inference transposition cache "
        "(shared across threads, rebuilt for each iteration's fresh network, "
        "always-replace eviction so memory stays bounded). 0 disables caching",
    )
    parser.add_argument(
        "--use-gumbel-search",
        action="store_true",
        default=False,
        help="Use Gumbel-Top-k + sequential halving root selection "
        "(MCTS::search_gumbel()) for self-play move selection instead of the "
        "default Dirichlet-noised PUCT search (MCTS::search())",
    )
    parser.add_argument(
        "--max-num-considered-actions",
        type=int,
        default=16,
        help="Gumbel search only: size m of the full search's Gumbel-Top-k "
        "candidate set. Raise (e.g. 32) while the network's policy prior is "
        "still too weak to rank the top moves reliably - sequential halving "
        "can never recover a move the initial Top-m cut excluded - and drop "
        "back toward 16 once it isn't. Fast searches derive their own m from "
        "this (capped at 8)",
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=replay_buffer_size,
        help="Replay buffer capacity in transitions. The default retains ~4 "
        "iterations of 1000-game self-play at ~80 plies/game (see the sizing "
        "comment above); with the sparse ReplayBuffer this is ~5KB/transition, "
        "so e.g. 500000 costs ~2.5GB",
    )
    parser.add_argument(
        "--replay-buffer-path",
        type=str,
        default=None,
        help="Persist the replay buffer to this file. At startup, if the file "
        "(and its '<path>.meta.json' sidecar) exists and its encoding signature "
        "matches this run, the buffer is preloaded so training skips the "
        "~15-20 min self-play refill ramp; otherwise the run starts empty. The "
        "buffer is re-saved every --replay-buffer-save-every iterations and once "
        "on graceful exit. Point a restart of the SAME config here for crash "
        "resilience; point a changed run here only when the encoding "
        "(game/arch/history) is unchanged - the guard REFUSES incompatible "
        "loads. NB the file is large (~state_bytes * capacity, e.g. ~8GB for a "
        "500k history-4 buffer); put it on fast local disk.",
    )
    parser.add_argument(
        "--replay-buffer-save-every",
        type=int,
        default=10,
        help="Iterations between replay-buffer snapshots when "
        "--replay-buffer-path is set (0 disables periodic saves, leaving only "
        "the on-exit save). Raise it to cut save I/O; lower it to lose fewer "
        "games to a hard crash.",
    )
    parser.add_argument(
        "--resignation-enabled",
        action="store_true",
        default=False,
        help="End decided self-play games early: a game is scored as a loss "
        "for the side to move once its search root value has read below "
        "--resignation-threshold on --resignation-consecutive-moves of its own "
        "successive turns (never before ply --resignation-min-ply). Keeps the "
        "replay buffer focused on contested positions instead of dead-lost "
        "endgame grinds",
    )
    parser.add_argument(
        "--resignation-threshold",
        type=float,
        default=-0.95,
        help="Side-to-move root value below which a turn counts toward the "
        "resignation streak",
    )
    parser.add_argument(
        "--resignation-consecutive-moves",
        type=int,
        default=3,
        help="How many of a side's successive turns must read below "
        "--resignation-threshold before it resigns (guards against one-off "
        "evaluation noise)",
    )
    parser.add_argument(
        "--resignation-min-ply",
        type=int,
        default=60,
        help="Earliest 0-based ply at which a turn may count toward a "
        "resignation streak",
    )
    parser.add_argument(
        "--resignation-disable-probability",
        type=float,
        default=0.1,
        help="Fraction of games that ignore resignation and play to the "
        "natural end, keeping the false-resignation rate observable "
        "(AlphaGo Zero's guard)",
    )
    parser.add_argument(
        "--fpu-reduction",
        type=float,
        default=0.0,
        help="First-play-urgency reduction: an unvisited MCTS child is scored "
        "at its parent's running value minus this amount, instead of the "
        "assume-draw 0.0. 0.1 matches the LC0/reference default; 0.0 keeps the "
        "original behavior",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory. Defaults to runs/<game>/<timestamp>. "
        "Pass an empty string to disable TensorBoard logging.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5,
        help="Maximum number of auto-named TensorBoard runs to keep per game "
        "(oldest are deleted). Only applies when --log-dir is left at its "
        "default; ignored if you pass an explicit --log-dir. 0 disables pruning.",
    )

    config_args, _ = parser.parse_known_args()
    if config_args.config:
        with open(config_args.config) as f:
            config_values = json.load(f)
        known_dests = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_values) - known_dests)
        if unknown_keys:
            raise ValueError(
                f"Unknown key(s) in config file '{config_args.config}': {unknown_keys}"
            )
        parser.set_defaults(**config_values)
        logging.info(
            f"Loaded training config from '{config_args.config}': {config_values}"
        )

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = get_args()

    if args.lr_lower_bound > args.initial_lr:
        raise ValueError(
            f"--lr-lower-bound ({args.lr_lower_bound:.2e}) must be <= --initial-lr "
            f"({args.initial_lr:.2e}); the LR decays from initial_lr down to the "
            f"lower bound, it never rises"
        )
    if args.network_arch == "chess_v2" and args.game != "chess":
        raise ValueError("--network-arch chess_v2 is chess-only")
    if args.chess_encoder_history is not None and args.network_arch != "chess_v2":
        raise ValueError(
            "--chess-encoder-history requires --network-arch chess_v2 (it sizes "
            "that architecture's input; the legacy net's input shape is fixed "
            "by --residual-blocks/--num-filters instead)"
        )
    if args.self_play_network and not os.path.isfile(args.self_play_network):
        raise ValueError(
            f"--self-play-network '{args.self_play_network}' does not exist - "
            f"the frozen generator engine must be present before training starts"
        )
    if args.self_play_encoder_history is not None and not args.self_play_network:
        raise ValueError(
            "--self-play-encoder-history only applies to a frozen-generator "
            "bootstrap; it requires --self-play-network"
        )

    manager = CheckpointManager(
        f"{args.game}_{args.checkpoint_stem}",
        PROJ_ROOT / args.checkpoint_dir / args.game,
        args.max_checkpoints,
        tensorrt_every=args.tensorrt_every,
    )

    def make_fresh_network():
        game_cls = Connect4 if args.game == "connect4" else Chess
        return get_network(
            game_cls,
            resblock_filter_size=args.num_filters,
            residual_block_count=args.residual_blocks,
            network_arch=args.network_arch,
            chess_encoder_history=args.chess_encoder_history,
        )

    if args.initial_network:
        if os.path.isfile(args.initial_network):
            logging.info(f"Initializing network from '{args.initial_network}'.")
            network = AlphaZeroNetwork.load_az_network(args.initial_network, device)
            if isinstance(network, AlphaZeroNetwork) and (
                len(network.residual_blocks),
                network.conv_in.out_channels,
            ) != (args.residual_blocks, args.num_filters):
                logging.warning(
                    f"Loaded checkpoint architecture "
                    f"({len(network.residual_blocks)} blocks, "
                    f"{network.conv_in.out_channels} filters) differs from the "
                    f"configured fresh-network architecture "
                    f"({args.residual_blocks} blocks, {args.num_filters} "
                    f"filters). The checkpoint's architecture is kept - the "
                    f"config values only apply to fresh networks."
                )
        else:
            logging.warning(
                f"Initial network file '{args.initial_network}' not found. Initializing a fresh AlphaZero network."
            )
            network = make_fresh_network()
    else:
        logging.info(
            "No initial network provided. Initializing a fresh AlphaZero network."
        )
        network = make_fresh_network()

    manager.add_checkpoint(network)

    logging.info(
        f"Starting AlphaZero training loop: {args.loop_iterations} outer iterations, {args.games_in_each_iteration} games per iter."
    )

    if args.game == "connect4":
        game_data = (Connect4, self_play_connect4)
        encoder = None  # Connect4's own default encoder; no history variant.
        self_play_encoder = None
    else:
        game_data = (Chess, self_play)
        encoder = get_chess_encoder(args.chess_encoder_history)
        # Only build a distinct generator encoder for a frozen-generator
        # bootstrap whose generator uses a different encoding than the trainee;
        # otherwise leave it None so the generator shares the trainee's encoder.
        self_play_encoder = (
            get_chess_encoder(args.self_play_encoder_history)
            if args.self_play_network
            and args.self_play_encoder_history != args.chess_encoder_history
            else encoder
        )
        self_play_value_encoder = (
            get_chess_encoder(args.self_play_value_encoder_history)
            if args.self_play_value_network
            else self_play_encoder
        )

    # Replay-buffer persistence compatibility signature. `tag` captures the
    # trainee's encoding (game + arch + history); state_shape is a secondary
    # defensive check when an explicit encoder is in play.
    replay_buffer_tag = compute_tag(
        args.game, args.network_arch, args.chess_encoder_history
    )
    replay_buffer_state_shape = (
        list(encoder.state_shape()) if encoder is not None else None
    )

    if args.log_dir == "":
        log_dir = None
    elif args.log_dir is not None:
        log_dir = args.log_dir
    else:
        runs_root = PROJ_ROOT / "runs" / args.game
        prune_old_runs(runs_root, args.max_runs)
        log_dir = str(runs_root / datetime.now().strftime("%Y%m%d-%H%M%S"))

    if log_dir is not None:
        logging.info(
            f"Logging to TensorBoard at '{log_dir}'. Run `tensorboard --logdir {log_dir}` to monitor training."
        )

    self_play_and_train_loop(
        checkpoint_manager=manager,
        network_type=AlphaZeroNetwork,
        network_device=device,
        game_data=game_data,
        trainer_factory=partial(
            get_trainer,
            initial_lr=args.initial_lr,
            lr_lower_bound=args.lr_lower_bound,
        ),
        loop_iterations=args.loop_iterations,
        games_in_each_iteration=args.games_in_each_iteration,
        replay_buffer_size=args.replay_buffer_size,
        training_iterations=args.training_iterations,
        minibatch_size=args.minibatch_size,
        batch_size=args.batch_size,
        thread_count=args.thread_count,
        max_moves=args.max_moves,
        mcts_batch_size=args.mcts_batch_size,
        mcts_simulations=args.mcts_simulations,
        fast_mcts_simulations=args.fast_mcts_simulations,
        full_search_probability=args.full_search_probability,
        transposition_cache_entries=args.transposition_cache_entries,
        use_gumbel_search=args.use_gumbel_search,
        max_num_considered_actions=args.max_num_considered_actions,
        resignation_enabled=args.resignation_enabled,
        resignation_threshold=args.resignation_threshold,
        resignation_consecutive_moves=args.resignation_consecutive_moves,
        resignation_min_ply=args.resignation_min_ply,
        resignation_disable_probability=args.resignation_disable_probability,
        fpu_reduction=args.fpu_reduction,
        self_play_network_path=args.self_play_network,
        self_play_value_network_path=args.self_play_value_network,
        encoder=encoder,
        self_play_encoder=self_play_encoder,
        self_play_value_encoder=self_play_value_encoder,
        replay_buffer_path=args.replay_buffer_path,
        replay_buffer_save_every=args.replay_buffer_save_every,
        replay_buffer_tag=replay_buffer_tag,
        replay_buffer_state_shape=replay_buffer_state_shape,
        log_dir=log_dir,
    )
