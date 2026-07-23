#ifndef DUAL_INFERER_HPP
#define DUAL_INFERER_HPP

#include "inferer.hpp"
#include <memory>
#include <vector>

// Inferer that takes legal_actions + logits from a policy network and value
// from a separate value network. The value network's policy output is discarded.
// Both networks are called for every batch; each has its own InferenceCache.
//
// When value_inferer is null this class is never constructed — callers use the
// policy inferer directly (the zero-overhead single-network path).
class DualNetworkInferer : public Inferer {
  public:
    DualNetworkInferer(std::unique_ptr<Inferer> policy_inferer,
                       std::unique_ptr<Inferer> value_inferer);

    std::vector<inference_result> infer(const std::vector<const GameState *> &states) override;

  private:
    std::unique_ptr<Inferer> policy_inferer_;
    std::unique_ptr<Inferer> value_inferer_;
};

#endif // DUAL_INFERER_HPP
