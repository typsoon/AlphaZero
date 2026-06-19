#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU.\n";
    } else {
        std::cout << "CUDA is not available. Using CPU.\n";
    }

    // Set device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Create a tensor and move it to the selected device
    torch::Tensor tensor = torch::rand({3, 3}).to(device);
    std::cout << "Tensor on device:\n" << tensor << std::endl;

    // Simple tensor operation
    torch::Tensor result = tensor * 2;
    std::cout << "Result after multiplication:\n" << result << std::endl;

    return 0;
}
