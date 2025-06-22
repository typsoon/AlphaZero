#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <torch/torch.h>

struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    ResidualBlockImpl(int channels);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResidualBlock);

struct AlphaZeroNetworkImpl : torch::nn::Module {
    int height, width;

    torch::nn::Conv2d conv_in{nullptr};
    torch::nn::BatchNorm2d bn_in{nullptr};
    torch::nn::Sequential residuals{nullptr};

    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::BatchNorm2d policy_bn{nullptr};
    torch::nn::Linear policy_fc{nullptr};

    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm2d value_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr}, value_fc2{nullptr};

    AlphaZeroNetworkImpl(int input_channels, int h, int w, int num_res_blocks,
                         int action_size, int num_filters);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(AlphaZeroNetwork);

#endif // NETWORK_HPP
