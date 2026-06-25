import argparse
import subprocess
import sys
import re
import ctypes
from pathlib import Path

try:
    # Preload libgomp.so.1 to prevent MKL "omp_get_num_procs" symbol lookup errors in conda environments
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib is not installed. Plotting will be skipped.")
    plt = None

def run_benchmark(benchmark_binary, network_path):
    print("Running C++ benchmark...")
    result = subprocess.run([benchmark_binary, network_path], check=True, capture_output=True, text=True)
    
    # Print the output so the user still sees the raw results
    print(result.stdout)
    
    if plt is None:
        return
        
    batches = []
    latencies = []
    evals_per_sec = []
    
    # Parse output, e.g.: Batch 32: 1.4834 ms latency, 21572.1 evals/sec
    pattern = re.compile(r"Batch\s+(\d+):\s+([\d.]+)\s+ms latency,\s+([\d.]+)\s+evals/sec")
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if match:
            batches.append(int(match.group(1)))
            latencies.append(float(match.group(2)))
            evals_per_sec.append(float(match.group(3)))
            
    if not batches:
        print("Could not parse benchmark output for plotting.")
        return
        
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Batch Size (log scale)')
    ax1.set_ylabel('Evals / sec', color=color)
    ax1.plot(batches, evals_per_sec, marker='o', color=color, linewidth=2, label='Evals / sec')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(batches)
    ax1.set_xticklabels([str(b) for b in batches])

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Latency (ms)', color=color)  
    ax2.plot(batches, latencies, marker='s', color=color, linewidth=2, linestyle='--', label='Latency (ms)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    fig.tight_layout()  
    plt.title('Inference Performance by Batch Size (CUDA)')
    plt.grid(True, alpha=0.3)
    
    plot_path = Path("performance_evaluation") / "benchmark_graph.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved benchmark graph to {plot_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark optimal batch size")
    parser.add_argument(
        "--inference-binary", required=True, help="Path to inference_server binary (used to find the benchmark binary)"
    )
    parser.add_argument(
        "--network-path", required=True, help="Path to neural network checkpoint"
    )
    args = parser.parse_args()

    benchmark_bin = Path(args.inference_binary).parent / "benchmark_batch_size"
    if not benchmark_bin.exists():
        print(f"Benchmark binary not found at {benchmark_bin}. Did you build the project?")
        sys.exit(1)
        
    run_benchmark(str(benchmark_bin), args.network_path)
