#include <chrono>
#include <connect4.hpp>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main(int argc, char *argv[]) { // NOLINT
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <network_path>\n";
        return 1;
    }
    std::string network_path = argv[1];

    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available, falling back to CPU\n";
        device = torch::Device(torch::kCPU);
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(network_path, device);
        module.eval();
    } catch (const c10::Error &e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    auto infer_method = module.get_method("infer");

    Connect4 game_cpu{torch::Device(torch::kCPU)};
    torch::Tensor state_cpu = std::move(game_cpu.get_canonical_state());

    std::vector<int> batch_sizes = {1,   2,    4,    8,    16,      32,      64,      128,    256,
                                    512, 1024, 2048, 4096, 1 << 13, 1 << 14, 1 << 15, 1 << 16};
    std::cout << "Benchmarking forward pass on device: " << (device.is_cuda() ? "CUDA" : "CPU")
              << "\n";

    for (int batch_size : batch_sizes) {
        // Stack the state to simulate batch ON CPU
        auto tensor_shape = state_cpu.sizes().vec();
        tensor_shape.insert(tensor_shape.begin(), batch_size);
        torch::Tensor batched_cpu = torch::empty(tensor_shape, state_cpu.options());
        for (int i = 0; i < batch_size; ++i) {
            batched_cpu[i].copy_(state_cpu);
        }

        // Measure memory transfer CPU -> GPU
        double mem_latency_ms = 0.0;
        torch::Tensor batched_gpu;
        if (device.is_cuda()) {
            // Warmup mem transfer
            batched_cpu.to(device);
            torch::cuda::synchronize();

            const int mem_iters = 20;
            auto mem_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < mem_iters; ++i) {
                batched_gpu = batched_cpu.to(device);
            }
            torch::cuda::synchronize();
            auto mem_end = std::chrono::high_resolution_clock::now();
            mem_latency_ms =
                std::chrono::duration<double, std::milli>(mem_end - mem_start).count() / mem_iters;
        } else {
            batched_gpu = batched_cpu;
        }

        // Warmup inference
        try {
            torch::NoGradGuard no_grad;
            for (int i = 0; i < 5; ++i) {
                infer_method({batched_gpu});
            }
            if (device.is_cuda()) {
                torch::cuda::synchronize();
            }
        } catch (const std::exception &e) {
            std::cerr << "Inference error: " << e.what() << "\n";
            return 1;
        }

        // Benchmark inference
        const int num_iterations = 20;
        auto start = std::chrono::high_resolution_clock::now();
        {
            torch::NoGradGuard no_grad;
            for (int i = 0; i < num_iterations; ++i) {
                infer_method({batched_gpu});
            }
            if (device.is_cuda()) {
                torch::cuda::synchronize();
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double latency_ms = duration_ms / num_iterations;
        double evals_per_sec = (batch_size * num_iterations) / (duration_ms / 1000.0);

        std::cout << "Batch " << batch_size << ": " << latency_ms << " ms latency, "
                  << mem_latency_ms << " ms memory transfer, " << evals_per_sec << " evals/sec\n";
    }
    return 0;
}
