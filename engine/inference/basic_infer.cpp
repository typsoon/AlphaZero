#include "basic_inferer.hpp"
#include "inferer.hpp"
#include <ATen/core/TensorBody.h>
#include <connect4.hpp>
#include <exception>
#include <filesystem>
#include <iostream>
#include <mutex>

#include <torch/torch.h> // For torch::Tensor and device
#include <utility>
#include <vector>

NetworkInferer::NetworkInferer(std::shared_ptr<Network> network,
                               torch::Device device)
    : Inferer(device), network(network),
      infer_method(network->get_method("infer")) {
  // Load the network weights here if needed, e.g.:
  // TODO: come back to this
  // torch::load(network, network_file_path);

  // network.to(device);
  // network.eval();
  // torch::jit::script::Module module = torch::jit::load(network_file_path);
}

// infer method implementation
vector<inference_result>
NetworkInferer::infer(std::vector<GameState> game_states) {
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

  auto result = infer_method({batched});
  auto outputs = result.toTuple()->elements();
  torch::Tensor policy = outputs[0].toTensor();
  torch::Tensor value = outputs[1].toTensor();

  std::vector<inference_result> out;
  out.reserve(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    out.push_back({policy[i], value[i].item<float>()});
  }

  return out;
}

typedef torch::jit::script::Module Network;
std::shared_ptr<Network> get_network_func(std::string network_file_path,
                                          torch::Device device) {
  if (std::filesystem::exists(network_file_path)) {
    return std::make_shared<Network>(
        torch::jit::load(network_file_path, device));
  } else {
    std::cout << "File " << network_file_path << " doesn't exist\n";
    throw std::exception();
  }
}

// Constructor for NetworkInfererFactory
NetworkInfererFactory::NetworkInfererFactory(
    const std::string &network_file_path, torch::Device device)
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
