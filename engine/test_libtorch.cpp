#include <sstream>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

int main() {
    spdlog::info("LibTorch version: {}", TORCH_VERSION);

    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        spdlog::info("CUDA is available! Using GPU.");
    } else {
        spdlog::info("CUDA is not available. Using CPU.");
    }

    // Set device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Create a tensor and move it to the selected device
    torch::Tensor tensor = torch::rand({3, 3}).to(device);
    std::stringstream ss_tensor;
    ss_tensor << tensor;
    spdlog::info("Tensor on device:\n{}", ss_tensor.str());

    // Simple tensor operation
    torch::Tensor result = tensor * 2;
    std::stringstream ss_result;
    ss_result << result;
    spdlog::info("Result after multiplication:\n{}", ss_result.str());

    return 0;
}
