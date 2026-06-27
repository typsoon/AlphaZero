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

class DynamicBatcher;

class NetworkInferer : public Inferer {
  private:
    std::shared_ptr<DynamicBatcher> batcher;

  public:
    NetworkInferer(std::shared_ptr<DynamicBatcher> batcher, torch::Device device);

    vector<inference_result> infer(const vector<const GameState *> &states) override;
};

class NetworkInfererFactory : public InfererFactory {
  private:
    using Network = torch::jit::script::Module;

    std::string network_file_path;
    torch::Device device;
    int wait_for_count;
    int timeout_ms;

    std::shared_ptr<Network> network;
    std::shared_ptr<DynamicBatcher> batcher;

    std::mutex get_inferer_mutex;

  public:
    NetworkInfererFactory(std::string network_file_path, torch::Device device,
                          int wait_for_count = 1, int timeout_ms = 10);

    std::unique_ptr<Inferer> get_inferer() override;
};

#endif // NETWORK_INFERER_HPP
