#!/usr/bin/env python3
"""Benchmark three inference paths for a chess net, on both compile time and
inference latency across batch sizes:

  1. scripted        - TorchScript (LibTorch), no TensorRT. Baseline.
  2. onnx-trt        - torch.onnx.export -> tensorrt.OnnxParser -> builder (FP16).
  3. torch-trt       - torch_tensorrt.compile(scripted, ir="torchscript") - the
                       project's actual .pt_trt route (network.tensorrt_and_save_network).

Both TRT engines are built with the SAME opt/max batch shapes so the inference
comparison is apples-to-apples. Pure Python; needs `onnx`, and the tensorrt /
cuda libs on LD_LIBRARY_PATH (same env as hist_run/cron/common.sh).

    python -m performance_evaluation.benchmark_onnx_trt \
        --network-path champions/chess_AZNetwork_20260718_1432.pt \
        --batch-sizes 1 32 256 512 --opt-batch 256
"""

import argparse
import os
import time
from pathlib import Path

import torch


def load_net(path: str, device: torch.device):
    """Load a v1 (AlphaZeroNetwork) or v2 (ChessAzV2Network) checkpoint; return
    (net.eval(), input_channels). AlphaZeroNetwork.load_az_network dispatches on
    the checkpoint's "architecture" field."""
    from python.network import AlphaZeroNetwork

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = AlphaZeroNetwork.load_az_network(path, device)
    return net.eval(), int(ckpt["init_args"]["input_channels"])


def build_onnx_trt(net, channels, opt_batch, max_batch, onnx_path, timing_cache_path=None):
    """torch.onnx.export -> TensorRT FP16 engine. Returns (engine, build_s).

    build_s times ONLY builder.build_serialized_network (the part a timing cache
    accelerates), not the onnx export/parse. If timing_cache_path is given, an
    existing cache there is loaded before the build and the (updated) cache is
    written back after - so a second call with the same path is warm-cache."""
    import tensorrt as trt

    dummy = torch.randn(opt_batch, channels, 8, 8, device=next(net.parameters()).device)
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=["input"], output_names=["policy", "value"],
        dynamic_axes={"input": {0: "b"}, "policy": {0: "b"}, "value": {0: "b"}},
        opset_version=17,
    )
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        raise RuntimeError(
            "ONNX->TRT parse failed:\n  "
            + "\n  ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    prof.set_shape("input", (1, channels, 8, 8), (opt_batch, channels, 8, 8), (max_batch, channels, 8, 8))
    cfg.add_optimization_profile(prof)

    tcache = None
    if timing_cache_path is not None:
        data = open(timing_cache_path, "rb").read() if os.path.exists(timing_cache_path) else b""
        tcache = cfg.create_timing_cache(data)
        cfg.set_timing_cache(tcache, ignore_mismatch=False)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, cfg)
    build_s = time.time() - t0
    if serialized is None:
        raise RuntimeError("TRT build returned None")
    if tcache is not None:
        with open(timing_cache_path, "wb") as f:
            f.write(tcache.serialize())
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    return engine, build_s


def build_torch_trt(net, opt_batch, max_batch, trt_path):
    """The project's route: torch_tensorrt.compile from TorchScript (via the
    net's own tensorrt_and_save_network). Returns (module, compile_s)."""
    import torch_tensorrt  # noqa: F401  (registers the TRT torchscript ops)

    t0 = time.time()
    net.tensorrt_and_save_network(
        trt_path, max_first_dim_of_input=max_batch, opt_first_dim_of_input=opt_batch
    )
    compile_s = time.time() - t0
    module = torch.jit.load(trt_path).eval()
    return module, compile_s


def time_trt_engine(engine, x, iters):
    """Time an ONNX-built TRT engine on cuda tensor x (uses torch tensors as the
    device I/O buffers)."""
    import tensorrt as trt

    ctx = engine.create_execution_context()
    ctx.set_input_shape("input", tuple(x.shape))
    outs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outs[name] = torch.empty(tuple(ctx.get_tensor_shape(name)), device=x.device)
    ctx.set_tensor_address("input", x.data_ptr())
    for name, t in outs.items():
        ctx.set_tensor_address(name, t.data_ptr())
    s = torch.cuda.current_stream()
    for _ in range(10):
        ctx.execute_async_v3(s.cuda_stream)
    s.synchronize()
    t0 = time.time()
    for _ in range(iters):
        ctx.execute_async_v3(s.cuda_stream)
    s.synchronize()
    return (time.time() - t0) / iters


def time_module(mod, x, iters):
    with torch.no_grad():
        for _ in range(10):
            mod(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            mod(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network-path", required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 256, 512])
    ap.add_argument("--opt-batch", type=int, default=256, help="TRT optimization-profile opt batch")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda")
    net, channels = load_net(args.network_path, device)
    max_batch = max(args.batch_sizes)
    opt_batch = min(args.opt_batch, max_batch)
    stem = Path(args.network_path).stem
    print(f"loaded {args.network_path} (channels={channels}); opt_batch={opt_batch} max_batch={max_batch}\n")

    scripted = torch.jit.script(net).eval()
    onnx_path = f"/tmp/bench_{stem}.onnx"
    cache_path = f"/tmp/bench_{stem}.trtcache"
    if os.path.exists(cache_path):
        os.remove(cache_path)  # start cold
    # Cold build populates+writes the timing cache; the warm build reuses it.
    # Weights don't affect kernel timing, so the warm case is exactly the
    # training per-iteration recompile (same architecture, new weights).
    onnx_engine, onnx_cold = build_onnx_trt(net, channels, opt_batch, max_batch, onnx_path, cache_path)
    _, onnx_warm = build_onnx_trt(net, channels, opt_batch, max_batch, onnx_path, cache_path)
    torch_trt_mod, ttrt_compile = build_torch_trt(net, opt_batch, max_batch, f"/tmp/bench_{stem}.pt_trt")

    print("=== COMPILE TIME (TRT build only) ===")
    print(f"  onnx-trt  cold cache : {onnx_cold:6.1f}s")
    print(f"  onnx-trt  WARM cache : {onnx_warm:6.1f}s  ({onnx_cold / onnx_warm:.1f}x faster - the per-iteration recompile case)")
    print(f"  torch-trt (no cache) : {ttrt_compile:6.1f}s\n")

    print("=== INFERENCE (ms/call, and speedup vs scripted) ===")
    print(f"{'batch':>6} {'scripted':>10} {'onnx-trt':>10} {'torch-trt':>10}   {'onnx x':>7} {'torch x':>7}")
    for b in args.batch_sizes:
        x = torch.randn(b, channels, 8, 8, device=device)
        ts = time_module(scripted, x, args.iters) * 1e3
        to = time_trt_engine(onnx_engine, x, args.iters) * 1e3
        tt = time_module(torch_trt_mod, x, args.iters) * 1e3
        print(f"{b:>6} {ts:>10.3f} {to:>10.3f} {tt:>10.3f}   {ts / to:>6.2f}x {ts / tt:>6.2f}x")


if __name__ == "__main__":
    main()
