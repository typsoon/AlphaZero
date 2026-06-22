#include "basic_inferer.hpp"
#include "inferer.hpp"
#include <ATen/core/TensorBody.h>
#include <connect4.hpp>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <spdlog/spdlog.h>

#include <torch/torch.h> // For torch::Tensor and device
#include <utility>
#include <vector>

NetworkInferer::NetworkInferer(std::shared_ptr<Network> network, torch::Device device)
    : Inferer(device), network(network), infer_method(network->get_method("infer")) {}

// infer method implementation
vector<inference_result> NetworkInferer::infer(std::vector<GameState> game_states) {
    if (game_states.empty()) {
        return {};
    }

    auto batch_size = game_states.size();
    vector<Tensor> game_states_tensors;
    game_states_tensors.reserve(batch_size);
    for (auto &&state : game_states) {
        game_states_tensors.emplace_back(std::move(state));
    }

    auto options = game_states_tensors[0].options();
    auto tensor_shape = game_states_tensors[0].sizes().vec();

    tensor_shape.insert(tensor_shape.begin(), batch_size);

    torch::Tensor batched = torch::empty(tensor_shape, options);

    for (int i = 0; i < batch_size; ++i) {
        batched[i].copy_(game_states_tensors[i]);
    }

    std::vector<inference_result> out;
    try {
        torch::NoGradGuard no_grad;
        auto result = infer_method({batched});
        auto outputs = result.toTuple()->elements();
        torch::Tensor policy = outputs[0].toTensor();
        torch::Tensor value = outputs[1].toTensor();

        out.reserve(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            out.push_back({policy[i], value[i].item<float>()});
        }
    } catch (const std::exception &e) {
        spdlog::error("Exception caught in NetworkInferer::infer: {}", e.what());
        throw;
    }

    return out;
}

typedef torch::jit::script::Module Network;
std::shared_ptr<Network> get_network_func(std::string network_file_path, torch::Device device) {
    if (std::filesystem::exists(network_file_path)) {
        try {
            return std::make_shared<Network>(torch::jit::load(network_file_path, device));
        } catch (const c10::Error &e) {
            spdlog::error(
                "Failed to load network from {}. Ensure it is exported using TorchScript.",
                network_file_path);
            throw std::runtime_error("Network file is not in TorchScript format");
        } catch (const std::exception &e) {
            spdlog::error("Failed to load network: {}", e.what());
            throw std::runtime_error("Failed to load network");
        }
    } else {
        spdlog::error("File {} doesn't exist", network_file_path);
        throw std::runtime_error("Network file not found");
    }
}

// Constructor for NetworkInfererFactory
NetworkInfererFactory::NetworkInfererFactory(const std::string &network_file_path,
                                             torch::Device device)
    : network_file_path(network_file_path), device(device),
      network(get_network_func(network_file_path, device)) {
    network->to(device);
    network->eval();
}

// get_inferer method implementation
std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    auto lock_guard = std::lock_guard(get_inferer_mutex);
    return std::make_unique<NetworkInferer>(network, device);
}
