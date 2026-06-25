#include <chrono>
#include <connect4.hpp>
#include <iostream>
#include <memory>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main(int argc, char *argv[]) {
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

    Connect4 game(device);
    torch::Tensor state = std::move(game.get_canonical_state());

    std::vector<int> batch_sizes = {1,   2,    4,    8,    16,      32,      64,      128,    256,
                                    512, 1024, 2048, 4096, 1 << 13, 1 << 14, 1 << 15, 1 << 16};
    std::cout << "Benchmarking forward pass on device: " << (device.is_cuda() ? "CUDA" : "CPU")
              << "\n";

    for (int batch_size : batch_sizes) {
        // Stack the state to simulate batch
        auto tensor_shape = state.sizes().vec();
        tensor_shape.insert(tensor_shape.begin(), batch_size);
        torch::Tensor batched = torch::empty(tensor_shape, state.options());
        for (int i = 0; i < batch_size; ++i) {
            batched[i].copy_(state);
        }

        // Warmup
        try {
            torch::NoGradGuard no_grad;
            for (int i = 0; i < 5; ++i) {
                infer_method({batched});
            }
            if (device.is_cuda()) {
                torch::cuda::synchronize();
            }
        } catch (const std::exception &e) {
            std::cerr << "Inference error: " << e.what() << "\n";
            return 1;
        }

        // Benchmark
        const int num_iterations = 20;
        auto start = std::chrono::high_resolution_clock::now();
        {
            torch::NoGradGuard no_grad;
            for (int i = 0; i < num_iterations; ++i) {
                infer_method({batched});
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
                  << evals_per_sec << " evals/sec\n";
    }
    return 0;
}
