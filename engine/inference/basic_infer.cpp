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

    torch::Tensor batched = torch::stack(game_states_tensors, 0);

    std::vector<inference_result> out;
    try {
        torch::NoGradGuard no_grad;
        auto result = infer_method({batched});
        auto outputs = result.toTuple()->elements();
        torch::Tensor policy = outputs[0].toTensor().cpu();
        torch::Tensor value = outputs[1].toTensor().cpu().contiguous();
        const float *value_ptr = value.data_ptr<float>();

        out.reserve(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            out.emplace_back(policy[i], value_ptr[i]);
        }
    } catch (const std::exception &e) {
        spdlog::error("Exception caught in NetworkInferer::infer: {}", e.what());
        throw;
    }

    return out;
}

vector<inference_result> NetworkInferer::infer(torch::Tensor batched) {
    if (batched.size(0) == 0) {
        return {};
    }

    auto batch_size = batched.size(0);
    batched = batched.to(device);

    std::vector<inference_result> out;
    try {
        torch::NoGradGuard no_grad;
        auto result = infer_method({batched});
        auto outputs = result.toTuple()->elements();
        torch::Tensor policy = outputs[0].toTensor().cpu();
        torch::Tensor value = outputs[1].toTensor().cpu().contiguous();
        const float *value_ptr = value.data_ptr<float>();

        out.reserve(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            out.emplace_back(policy[i], value_ptr[i]);
        }
    } catch (const std::exception &e) {
        spdlog::error("Exception caught in NetworkInferer::infer: {}", e.what());
        throw;
    }

    return out;
}

using Network = torch::jit::script::Module;
static std::shared_ptr<Network> get_network_func(std::string network_file_path,
                                                 torch::Device device) {
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
NetworkInfererFactory::NetworkInfererFactory(std::string network_file_path,
                                             torch::Device device)
    : network_file_path(std::move(network_file_path)), device(device),
      network(get_network_func(this->network_file_path, device)) {
    network->to(device);
    network->eval();
}

// get_inferer method implementation
std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    auto lock_guard = std::scoped_lock(get_inferer_mutex);
    return std::make_unique<NetworkInferer>(network, device);
}
