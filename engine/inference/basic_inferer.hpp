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
#include <unordered_map>
#include <utility>

// Forward declaration of AlphaZeroNetworkImpl (if needed)
// class AlphaZeroNetworkImpl;

typedef std::pair<torch::Tensor, float> inference_result;

class NetworkInferer : public Inferer {
  private:
    typedef torch::jit::Method Method;
    typedef torch::jit::script::Module Network;

    // torch::jit::script::Module network;
    // torch::jit::script::Method infer_method;
    std::shared_ptr<Network> network;
    Method infer_method;

  public:
    NetworkInferer(std::shared_ptr<Network> method, torch::Device device);

    vector<inference_result>
    infer(vector<GameState> game_state_tensor) override;
    // Optional: expose device if needed
    // torch::Device device() const;
};

class NetworkInfererFactory : public InfererFactory {
  private:
    typedef torch::jit::Method Method;
    typedef torch::jit::script::Module Network;

    std::string network_file_path;
    torch::Device device;

    std::shared_ptr<Network> network;

    std::mutex get_inferer_mutex;

  public:
    NetworkInfererFactory(const std::string &network_file_path,
                          torch::Device device);

    std::unique_ptr<Inferer> get_inferer() override;
};

#endif // NETWORK_INFERER_HPP
