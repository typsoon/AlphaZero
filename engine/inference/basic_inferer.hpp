#ifndef NETWORK_INFERER_HPP
#define NETWORK_INFERER_HPP

#include "game.hpp"
#include "inferer.hpp"
#include <connect4.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <utility>

// Forward declaration of AlphaZeroNetworkImpl (if needed)
// class AlphaZeroNetworkImpl;

using inference_result = std::pair<torch::Tensor, float>;

class NetworkInferer : public Inferer {
  private:
    using Method = torch::jit::Method;
    using Network = torch::jit::script::Module;

    // torch::jit::script::Module network;
    // torch::jit::script::Method infer_method;
    std::shared_ptr<Network> network;
    Method infer_method;

  public:
    NetworkInferer(std::shared_ptr<Network> method, torch::Device device);

    vector<inference_result> infer(vector<GameState> game_state_tensor) override;
    vector<inference_result> infer(torch::Tensor batched_states) override;
    // Optional: expose device if needed
    // torch::Device device() const;
};

class NetworkInfererFactory : public InfererFactory {
  private:
    using Method = torch::jit::Method;
    using Network = torch::jit::script::Module;

    std::string network_file_path;
    torch::Device device;

    std::shared_ptr<Network> network;

    std::mutex get_inferer_mutex;

  public:
    NetworkInfererFactory(std::string network_file_path, torch::Device device);

    std::unique_ptr<Inferer> get_inferer() override;
};

#endif // NETWORK_INFERER_HPP
