#include "dual_inferer.hpp"

DualNetworkInferer::DualNetworkInferer(std::unique_ptr<Inferer> policy_inferer,
                                       std::unique_ptr<Inferer> value_inferer)
    : Inferer(policy_inferer->device), policy_inferer_(std::move(policy_inferer)),
      value_inferer_(std::move(value_inferer)) {}

std::vector<inference_result>
DualNetworkInferer::infer(const std::vector<const GameState *> &states) {
    auto results = policy_inferer_->infer(states);
    auto value_results = value_inferer_->infer(states);
    for (size_t i = 0; i < results.size(); ++i)
        results[i].value = value_results[i].value;
    return results;
}
