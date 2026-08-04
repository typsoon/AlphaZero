from os import PathLike, fspath
from typing import Tuple
import torch

import torch.nn as nn
import torch.nn.functional as F


def _onnx_to_trt_engine(
    module,
    in_channels,
    height,
    width,
    opt_batch,
    max_batch,
    out_path,
    timing_cache_path=None,
):
    """Compile `module` to a TensorRT engine via ONNX (torch.onnx.export ->
    tensorrt.OnnxParser -> builder, FP16) and write the RAW serialized engine to
    out_path. Unlike the torch_tensorrt path this produces a plain TRT engine
    (not a TorchScript-wrapped module), so its output must be loaded by a
    TensorRT runtime, not torch.jit.load. Needs `onnx` and `tensorrt`, but NOT
    torch_tensorrt. If timing_cache_path is given the cache is loaded before and
    rewritten after the build - a warm cache makes repeated builds of the same
    architecture (e.g. the per-iteration recompile) much faster.

    TRT 10.x and earlier ("weak typing") pick FP16 kernels via a builder flag on
    an FP32 graph. TRT 11 removed that flag entirely - networks are always
    "strongly typed" and take precision straight from the ONNX graph's own
    tensor dtypes, so FP16 has to come from exporting an actual FP16 module. The
    version check below picks whichever the installed `tensorrt` supports, so
    this one function works unchanged across both TRT generations. The
    strongly-typed branch exports a deep copy cast to .half() rather than the
    live `module` in place, since `module` is normally the trainer's actual
    in-memory network and mutating its dtype would desync it from its
    optimizer's per-parameter state on the next training step."""
    import copy
    import os

    import tensorrt as trt

    weak_typing = hasattr(trt.BuilderFlag, "FP16")

    device = next(module.parameters()).device
    export_module = module if weak_typing else copy.deepcopy(module).half()
    dummy = torch.randn(
        opt_batch,
        in_channels,
        height,
        width,
        device=device,
        dtype=torch.float32 if weak_typing else torch.float16,
    )
    onnx_path = fspath(out_path) + ".onnx"
    torch.onnx.export(
        export_module,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "b"}, "policy": {0: "b"}, "value": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if weak_typing:
        net = builder.create_network()
    else:
        net = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        errs = "\n  ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError("ONNX->TRT parse failed:\n  " + errs)
    cfg = builder.create_builder_config()
    if weak_typing:
        cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    prof.set_shape(
        "input",
        (1, in_channels, height, width),
        (opt_batch, in_channels, height, width),
        (max_batch, in_channels, height, width),
    )
    cfg.add_optimization_profile(prof)
    tcache = None
    if timing_cache_path is not None:
        data = (
            open(timing_cache_path, "rb").read()
            if os.path.exists(timing_cache_path)
            else b""
        )
        tcache = cfg.create_timing_cache(data)
        cfg.set_timing_cache(tcache, ignore_mismatch=False)
    serialized = builder.build_serialized_network(net, cfg)
    if serialized is None:
        raise RuntimeError("TRT build (build_serialized_network) returned None")
    if tcache is not None:
        with open(timing_cache_path, "wb") as f:
            f.write(tcache.serialize())
    with open(fspath(out_path), "wb") as f:
        f.write(bytes(serialized))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, X):
        residual = X
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        X += residual
        return F.relu(X)


class AlphaZeroNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        height: int,
        width: int,
        num_residual_blocks: int,
        action_size: int,
        num_filters: int,
    ):
        super().__init__()

        self._height = height
        self._width = width

        self.conv_in = nn.Conv2d(
            input_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        self.bn_in = nn.BatchNorm2d(num_filters)

        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * height * width, action_size)

        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(height * width, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

    def forward(self, X):
        X = F.relu(self.bn_in(self.conv_in(X)))
        X = self.residual_blocks(X)

        policy = F.relu(self.policy_bn(self.policy_conv(X)))
        policy = policy.flatten(start_dim=1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(X)))
        value = value.flatten(start_dim=1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    @torch.jit.export
    def infer(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and return policy and value.
        This method is exposed for TorchScript.
        """
        return self.forward(X)

    def save_az_network(self, path: PathLike):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "init_args": {
                    "input_channels": self.conv_in.in_channels,
                    "height": self._height,
                    "width": self._width,
                    "num_residual_blocks": len(self.residual_blocks),
                    "action_size": self.policy_fc.out_features,
                    "num_filters": self.conv_in.out_channels,
                },
            },
            fspath(path),
        )

    def script_and_save_network(self, path: PathLike):
        scripted_model = torch.jit.script(self)
        scripted_model.save(fspath(path))

    def tensorrt_and_save_network(
        self,
        path: PathLike,
        max_first_dim_of_input=4096,
        opt_first_dim_of_input=1024,
        backend: str = "torch_tensorrt",
        timing_cache_path=None,
    ):
        # backend: "torch_tensorrt" (default) compiles the TorchScript IR into a
        # TorchScript-wrapped .pt_trt that the C++ inference server loads via
        # libtorch. "onnx" exports to ONNX and builds a RAW serialized TRT engine
        # (faster to compile, esp. with a warm timing_cache_path, and needs no
        # torch_tensorrt) - but its output must be loaded by a TensorRT runtime,
        # not torch.jit.load. See _onnx_to_trt_engine.
        # TensorRT compilation requires eval mode and cuda. Restore the
        # caller's original device/mode afterward, even on failure - this gets
        # called mid-training on the live model (e.g. from
        # CheckpointManager.add_checkpoint), and the device shift previously
        # had no way back: unlike .eval() (undone by the next
        # AlphaZeroTrainer.train() call), a CPU->CUDA move here stuck
        # permanently. No-op in the actual training setup, which already runs
        # on CUDA throughout - this only matters for CPU-mode training/tests.
        was_training = self.training
        original_device = next(self.parameters()).device

        try:
            self.eval()
            if not next(self.parameters()).is_cuda:
                self.cuda()

            if backend == "onnx":
                _onnx_to_trt_engine(
                    self,
                    self.conv_in.in_channels,
                    self._height,
                    self._width,
                    opt_first_dim_of_input,
                    max_first_dim_of_input,
                    path,
                    timing_cache_path=timing_cache_path,
                )
                return
            if backend != "torch_tensorrt":
                raise ValueError(
                    f"unknown backend {backend!r} (use 'torch_tensorrt' or 'onnx')"
                )
            import torch_tensorrt  # noqa: F811

            # We need to provide input shape constraints for dynamic batch sizes.
            # DynamicBatcher's actual batch size isn't bounded by wait_for_count
            # the way it might look - while the worker thread is busy running one
            # inference, every other self-play thread keeps calling submit() and
            # piling more work into the next batch, so real batch sizes scale
            # with thread_count, not with wait_for_count. Measured directly (by
            # temporarily logging DynamicBatcher::process_batch's total_count over
            # a real chess self-play run) on a 12-core dev machine: at
            # thread_count=12 (not oversubscribed - 1 thread/core, the config
            # that best isolates real per-thread submission behavior instead of
            # OS-scheduler contention noise) batch sizes were median=205,
            # p99=369, max=498 - opt_shape=32 (this function's old default,
            # sized for an old, much smaller mcts_batch_size) undershoots that by
            # ~6x. opt_first_dim_of_input=1024/max_first_dim_of_input=4096 below
            # extrapolate that measurement linearly to a 64-core/thread_count=64
            # target machine (64/12 * median=205 =~ 1093, rounded down to 1024;
            # max scaled =~ 2656, given ~1.5x headroom to 4096 for burstiness
            # beyond the linear estimate) - this is an extrapolation, not a
            # direct measurement on that hardware, since this dev box only has
            # 12 cores. Re-measure with the same instrumentation on the real
            # training machine and re-generate if these turn out off.
            inputs = [
                torch_tensorrt.Input(
                    min_shape=[1, self.conv_in.in_channels, self._height, self._width],
                    opt_shape=[
                        opt_first_dim_of_input,
                        self.conv_in.in_channels,
                        self._height,
                        self._width,
                    ],
                    max_shape=[
                        max_first_dim_of_input,
                        self.conv_in.in_channels,
                        self._height,
                        self._width,
                    ],
                    dtype=torch.float32,
                )
            ]

            # NB (2026-07-16): a dynamo-IR + persistent-timing-cache variant was
            # tried here to cut the per-checkpoint recompile (~32s -> ~8s warm).
            # REVERTED: although it compiled faster and played equal in an arena,
            # the dynamo engine (saved via output_format="torchscript") ran
            # INFERENCE ~2x slower at every batch size (benchmarked 1..1024), and
            # inference dominates the loop - a net loss. The torchscript IR
            # produces the tighter/faster TRT engine, so keep it.
            scripted_model = torch.jit.script(self)
            # Dedicate total VRAM minus a 5GB safety buffer for PyTorch/OS (min 1GB)
            dyn_workspace = max(
                1 << 30, torch.cuda.get_device_properties(0).total_memory - (5 << 30)
            )
            trt_model = torch_tensorrt.compile(
                scripted_model,
                inputs=inputs,
                enabled_precisions={torch.float32, torch.float16},
                ir="torchscript",
                workspace_size=dyn_workspace,
            )
            trt_model.save(path)
        finally:
            self.to(original_device)
            if was_training:
                self.train()

    @staticmethod
    def load_az_network(path: PathLike, device: torch.device) -> "nn.Module":
        # Dispatches on the checkpoint's explicit "architecture" tag (borrowed
        # from the engine-zoo reference: never guess a checkpoint's format from
        # tensor shapes). Legacy checkpoints predate the tag and have none.
        checkpoint = torch.load(fspath(path), map_location=device, weights_only=True)
        architecture = checkpoint.get("architecture", "legacy")
        if architecture == "chess-v2":
            model: nn.Module = ChessAzV2Network(**checkpoint["init_args"])
        elif architecture == "legacy":
            model = AlphaZeroNetwork(**checkpoint["init_args"])
        else:
            raise ValueError(f"unknown checkpoint architecture tag: {architecture!r}")
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# --- Chess AlphaZero v2 (port of the engine-zoo Rust reference) --------------
#
# Architecture (mirrors crates/algorithms/src/alphazero/network/chess_v2.rs):
# 12 SE (squeeze-excitation) residual blocks x 128 channels, a spatial 73-plane
# policy head (the LC0/AZ chess move encoding: 56 queen-like + 8 knight + 9
# underpromotion planes per from-square), and a WDL (win/draw/loss) value head
# whose scalar for MCTS is P(win) - P(loss).
#
# Two deliberate adaptations to THIS repo's pipeline (engine stays untouched):
#  * Input is our engine's 19-plane canonical state (the reference uses
#    14*history+7 planes; adopting its history stacking would require C++
#    engine changes and is out of scope here).
#  * The engine's flat action space is (from*64+to)*5+promo (20480 ids) in
#    ABSOLUTE board coordinates, while write_canonical_state() row-flips the
#    board for black. The spatial policy is therefore produced in the
#    CANONICAL frame and translated to engine action ids inside the network
#    via two precomputed index maps (white=identity, black=row-flipped),
#    selected per sample from the side-to-move input plane (plane 12, which is
#    all-1 for white / all-0 for black). Distinct legal moves of one position
#    never collide on a spatial slot (queen-promotions share the queen-move
#    plane with non-promotion moves exactly like LC0, but a (from,to) pair is
#    either a promotion square or not - both ids are never legal at once).


_CHESS_V2_DIRECTIONS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
_CHESS_V2_KNIGHT_MOVES = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]
# Engine promotion codes (chess.hpp): 0=none, 1=queen, 2=rook, 3=knight,
# 4=bishop. Queen promotions ride the ordinary queen-move planes; the three
# underpromotions get dedicated planes in LC0's N, B, R order.
_CHESS_V2_UNDERPROMO_INDEX = {3: 0, 4: 1, 2: 2}

# engine-zoo's OWN plane geometry (crates/alphazero/src/representation/
# chess_v2.rs: KNIGHTS array and plane()'s underpromotion `m` match). Same 8
# knight deltas and 3 underpromotion direction groups as this repo's own
# arrays above, but in a DIFFERENT index order - required (together with
# ENGINE_ZOO_FLIPS_WHITE below) to correctly serve weights transplanted from
# engine-zoo (e.g. via convert_safetensors_v2.py); using this repo's own order
# against transplanted weights silently scrambles knight-move and
# underpromotion policy logits. See [[chess-v2-network-port]].
_CHESS_V2_KNIGHT_MOVES_ENGINE_ZOO = [
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
]
_CHESS_V2_UNDERPROMO_GROUP_ENGINE_ZOO = {0: 0, -1: 1, 1: 2}  # keyed by dc


def _chess_v2_plane(dr: int, dc: int, promo: int, knight_moves, underpromo_group):
    """Returns the 73-plane index for a CANONICAL-frame move delta, or None if
    the (delta, promo) combination is geometrically impossible (such action ids
    exist in the flat 20480 space but can never be legal). knight_moves/
    underpromo_group select this repo's own plane order or engine-zoo's (see
    _build_chess_v2_action_maps)."""
    if promo in _CHESS_V2_UNDERPROMO_INDEX:
        # In the canonical frame the side to move always advances toward row 0.
        if dr == -1 and -1 <= dc <= 1:
            return 64 + underpromo_group(dc) * 3 + _CHESS_V2_UNDERPROMO_INDEX[promo]
        return None
    if (dr, dc) in knight_moves:
        return 56 + knight_moves.index((dr, dc))
    distance = max(abs(dr), abs(dc))
    if distance == 0 or distance > 7:
        return None
    if dr == 0 or dc == 0 or abs(dr) == abs(dc):
        direction = ((dr > 0) - (dr < 0), (dc > 0) - (dc < 0))
        return _CHESS_V2_DIRECTIONS.index(direction) * 7 + (distance - 1)
    return None


def _build_chess_v2_action_maps(
    convention: str = "native",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Maps every flat engine action id to its canonical-frame spatial policy
    slot (plane*64 + from_square), once for white to move and once for black.
    Impossible ids alias slot 0; they are never legal so their logit is never
    read.

    convention="native" (default): this repo's own choice - white=identity
    geometry, black=rows flipped (matching ChessEncoderV2History's default
    flip_white=False / ChessEncoderV1) - correct for a network THIS repo
    trains itself.

    convention="engine_zoo": the reference implementation's own knight-move
    and underpromotion-direction plane order (see the ENGINE_ZOO arrays
    above) - required to correctly serve weights transplanted from
    engine-zoo. Despite engine-zoo's OWN source flipping White's rank index
    (not Black's), its rank index is numbered OPPOSITE to this engine's row
    axis (rank_index=0 -> rank "1", vs. this engine's row=0 -> rank "8" -
    confirmed via engine-zoo's crates/games/src/chess/notation.rs:
    `'1' + rank.to_index()`), so translated into THIS engine's row axis,
    engine-zoo's convention is white=identity/black=flipped - the SAME
    row-flip assignment as "native". Only the plane geometry differs; pair
    with the default ChessEncoderV2History(flip_white=False), NOT True."""
    if convention == "native":
        knight_moves = _CHESS_V2_KNIGHT_MOVES
        underpromo_group = lambda dc: dc + 1  # noqa: E731
    elif convention == "engine_zoo":
        knight_moves = _CHESS_V2_KNIGHT_MOVES_ENGINE_ZOO
        underpromo_group = lambda dc: _CHESS_V2_UNDERPROMO_GROUP_ENGINE_ZOO[dc]  # noqa: E731
    else:
        raise ValueError(f"unknown chess-v2 action_convention: {convention!r}")
    # Row-flip assignment is the SAME for both conventions - see docstring.
    white_flips = False

    map_white = torch.zeros(64 * 64 * 5, dtype=torch.long)
    map_black = torch.zeros(64 * 64 * 5, dtype=torch.long)
    for from_sq in range(64):
        r1, c1 = divmod(from_sq, 8)
        for to_sq in range(64):
            r2, c2 = divmod(to_sq, 8)
            for promo in range(5):
                action_id = (from_sq * 64 + to_sq) * 5 + promo
                dr, dc = r2 - r1, c2 - c1
                flipped_dr = (7 - r2) - (7 - r1)

                white_dr = flipped_dr if white_flips else dr
                white_row = (7 - r1) if white_flips else r1
                plane = _chess_v2_plane(white_dr, dc, promo, knight_moves, underpromo_group)
                if plane is not None:
                    map_white[action_id] = plane * 64 + white_row * 8 + c1

                black_dr = dr if white_flips else flipped_dr
                black_row = r1 if white_flips else (7 - r1)
                plane = _chess_v2_plane(black_dr, dc, promo, knight_moves, underpromo_group)
                if plane is not None:
                    map_black[action_id] = plane * 64 + black_row * 8 + c1
    return map_white, map_black


class SeResidualBlock(nn.Module):
    """conv-BN-relu-conv-BN with an LC0-style squeeze-excitation gate: the
    pooled block output produces a per-channel sigmoid scale and additive bias,
    applied before the residual add."""

    def __init__(self, channels: int, se_hidden: int):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se_reduce = nn.Linear(channels, se_hidden)
        self.se_expand = nn.Linear(se_hidden, 2 * channels)

    def forward(self, X):
        Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
        se = self.se_expand(F.relu(self.se_reduce(Y.mean(dim=[2, 3]))))
        scale = (
            torch.sigmoid(se.narrow(1, 0, self.channels)).unsqueeze(-1).unsqueeze(-1)
        )
        bias = se.narrow(1, self.channels, self.channels).unsqueeze(-1).unsqueeze(-1)
        return F.relu(X + scale * Y + bias)


class ChessAzV2Network(nn.Module):
    CHANNELS = 128
    RESIDUAL_BLOCKS = 12
    SE_HIDDEN = 16
    POLICY_PLANES = 73

    def __init__(
        self,
        input_channels: int,
        height: int = 8,
        width: int = 8,
        action_size: int = 64 * 64 * 5,
        stm_plane_index: int = 12,
        action_convention: str = "native",
    ):
        super().__init__()
        assert height == 8 and width == 8, "chess-v2 is an 8x8 chess-only network"
        assert action_size == 64 * 64 * 5, (
            "chess-v2 expects the engine's flat action space"
        )
        self._height = height
        self._width = width
        # Which input plane is the constant side-to-move indicator (1.0 white
        # to move / 0.0 black), used below to pick the policy's color frame.
        # Defaults to 12 (ChessEncoderV1's layout: 12 piece planes then
        # side-to-move). A history-stacked encoder (ChessEncoderV2History)
        # puts it later - at history*14 - so the caller building the network
        # for that encoder must pass the matching index; see injectors.py.
        self.stm_plane_index = stm_plane_index
        # "native" (default) or "engine_zoo" - see _build_chess_v2_action_maps.
        # Must be paired with the matching ChessEncoderV2History(flip_white=...)
        # at inference/self-play time; this class has no way to enforce that
        # pairing itself, since the encoder lives on the C++ side.
        self.action_convention = action_convention
        # Read by AlphaZeroTrainer to pick the WDL cross-entropy value loss.
        self.value_head = "wdl"

        c = self.CHANNELS
        self.stem_conv = nn.Conv2d(
            input_channels, c, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(c)
        self.blocks = nn.Sequential(
            *[SeResidualBlock(c, self.SE_HIDDEN) for _ in range(self.RESIDUAL_BLOCKS)]
        )
        self.policy_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(c)
        self.policy_out = nn.Conv2d(c, self.POLICY_PLANES, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2d(c, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * height * width, 128)
        self.value_fc2 = nn.Linear(128, 3)

        map_white, map_black = _build_chess_v2_action_maps(action_convention)
        self.register_buffer("_map_white", map_white)
        self.register_buffer("_map_black", map_black)

        self._init_weights()

    def _init_weights(self) -> None:
        """Deep-ResNet-from-scratch initialization (not from the AlphaZero papers,
        which never specify one - this follows the standard recipe used by strong
        open engines like KataGo/Leela): He-normal on ReLU convs/linears, then two
        corrections for a 12-block tower starting cold:
          * residual-identity init - zero the last BN gamma AND the SE additive
            branch (se_expand) in every block, so each SeResidualBlock starts as
            an exact identity map (X + scale*Y + bias with Y=0 and bias=0);
          * small-init the policy/value heads so early logits are near-uniform,
            avoiding a confidently-miscalibrated value head at step 0.
        Only affects freshly-built networks; loading a checkpoint overwrites all
        of this via state_dict, so existing nets are unchanged."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Residual-identity: each block contributes scale*Y + bias to X; zeroing
        # bn2's gamma makes Y=0 and zeroing se_expand makes the additive bias 0.
        for block in self.blocks:
            nn.init.zeros_(block.bn2.weight)
            nn.init.zeros_(block.se_expand.weight)
            nn.init.zeros_(block.se_expand.bias)
        # Near-uniform heads at init (policy_out and the WDL value_fc2).
        nn.init.normal_(self.policy_out.weight, std=0.01)
        nn.init.zeros_(self.policy_out.bias)
        nn.init.normal_(self.value_fc2.weight, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)

    def _heads(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the trunk and both heads: flat policy logits [B, 20480]
        (engine action ids) and raw WDL logits [B, 3]."""
        Y = F.relu(self.stem_bn(self.stem_conv(X)))
        Y = self.blocks(Y)

        policy = self.policy_out(F.relu(self.policy_bn(self.policy_conv(Y))))
        flat = policy.flatten(start_dim=1)  # [B, 73*64] canonical slots
        as_white = flat.index_select(1, self._map_white)
        as_black = flat.index_select(1, self._map_black)
        # Side-to-move plane: all ones for white, all zeros for black.
        stm = X[:, self.stm_plane_index, 0, 0].unsqueeze(1)
        policy_flat = as_white * stm + as_black * (1.0 - stm)

        value = F.relu(self.value_bn(self.value_conv(Y)))
        value = F.relu(self.value_fc1(value.flatten(start_dim=1)))
        wdl = self.value_fc2(value)
        return policy_flat, wdl

    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        """The deployed C++ contract (basic_infer binds get_method("forward"),
        and torch_tensorrt compiles exactly this method): flat policy logits
        and the scalar value MCTS consumes, P(win) - P(loss) in [-1, 1]."""
        policy_flat, wdl = self._heads(X)
        probabilities = torch.softmax(wdl, dim=1)
        value = probabilities.narrow(1, 0, 1) - probabilities.narrow(1, 2, 1)
        return policy_flat, value

    @torch.jit.export
    def infer(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias of forward() for API parity with AlphaZeroNetwork."""
        return self.forward(X)

    def train_outputs(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training interface: flat policy logits and RAW WDL logits, for the
        trainer's cross-entropy value loss (see AlphaZeroTrainer)."""
        return self._heads(X)

    def save_az_network(self, path: PathLike):
        torch.save(
            {
                "architecture": "chess-v2",
                "model_state_dict": self.state_dict(),
                "init_args": {
                    "input_channels": self.stem_conv.in_channels,
                    "height": self._height,
                    "width": self._width,
                    "action_size": 64 * 64 * 5,
                    "stm_plane_index": self.stm_plane_index,
                    "action_convention": self.action_convention,
                },
            },
            fspath(path),
        )

    def script_and_save_network(self, path: PathLike):
        scripted_model = torch.jit.script(self)
        scripted_model.save(fspath(path))

    def tensorrt_and_save_network(
        self,
        path: PathLike,
        max_first_dim_of_input=4096,
        opt_first_dim_of_input=1024,
        backend: str = "torch_tensorrt",
        timing_cache_path=None,
    ):
        # Same compile path and rationale as AlphaZeroNetwork's method (see the
        # comments there, incl. the torchscript-IR-over-dynamo NB and the
        # "onnx" vs "torch_tensorrt" backend note); kept separate instead of
        # refactored so the legacy class powering the live training run stays
        # byte-identical.
        was_training = self.training
        original_device = next(self.parameters()).device

        try:
            self.eval()
            if not next(self.parameters()).is_cuda:
                self.cuda()

            if backend == "onnx":
                _onnx_to_trt_engine(
                    self,
                    self.stem_conv.in_channels,
                    self._height,
                    self._width,
                    opt_first_dim_of_input,
                    max_first_dim_of_input,
                    path,
                    timing_cache_path=timing_cache_path,
                )
                return
            if backend != "torch_tensorrt":
                raise ValueError(
                    f"unknown backend {backend!r} (use 'torch_tensorrt' or 'onnx')"
                )
            import torch_tensorrt  # noqa: F811

            inputs = [
                torch_tensorrt.Input(
                    min_shape=[
                        1,
                        self.stem_conv.in_channels,
                        self._height,
                        self._width,
                    ],
                    opt_shape=[
                        opt_first_dim_of_input,
                        self.stem_conv.in_channels,
                        self._height,
                        self._width,
                    ],
                    max_shape=[
                        max_first_dim_of_input,
                        self.stem_conv.in_channels,
                        self._height,
                        self._width,
                    ],
                    dtype=torch.float32,
                )
            ]

            scripted_model = torch.jit.script(self)
            # Dedicate total VRAM minus a 5GB safety buffer for PyTorch/OS (min 1GB)
            dyn_workspace = max(
                1 << 30, torch.cuda.get_device_properties(0).total_memory - (5 << 30)
            )
            trt_model = torch_tensorrt.compile(
                scripted_model,
                inputs=inputs,
                enabled_precisions={torch.float32, torch.float16},
                ir="torchscript",
                # The int64 _map_white/_map_black index buffers can't be frozen
                # into TRT constants as-is; this truncates them to int32 inside
                # the engine (safe: slot indices max out at 73*64-1).
                truncate_long_and_double=True,
                workspace_size=dyn_workspace,
            )
            trt_model.save(path)
        finally:
            self.to(original_device)
            if was_training:
                self.train()
