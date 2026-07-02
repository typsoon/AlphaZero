#ifndef INFERER_H
#define INFERER_H

#include "game.hpp"
#include <torch/torch.h>

struct inference_result {
    // legal_actions[j] is the action index whose logit is legal_action_logits[j].
    // Inferer computes this itself (via GameState::get_legal_actions()) and hands
    // it back so callers don't need to separately request or recompute it. Only
    // these values are extracted, not the full dense policy row - Chess's action
    // space is 20480-wide but a position has ~30-40 legal moves, so this avoids
    // copying/retaining ~500x more data than any consumer actually reads.
    std::vector<int> legal_actions;
    std::vector<float> legal_action_logits;
    float value;
};
struct Inferer {
    // Inferer should have a method to predict the policy and value for a given
    // game state.
    virtual std::vector<inference_result> infer(const std::vector<const GameState *> &states) = 0;
    torch::Device device;
    virtual ~Inferer() = default;

    Inferer(torch::Device device) : device(device) {}
};

class InfererFactory {
  public:
    virtual ~InfererFactory() = default;

    // pure virtual function - like @abstractmethod in Python
    virtual std::unique_ptr<Inferer> get_inferer() = 0;
};

#endif // !INFERER_H
