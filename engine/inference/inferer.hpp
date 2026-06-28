#ifndef INFERER_H
#define INFERER_H

#include "game.hpp"
#include <torch/torch.h>

struct inference_result {
    torch::Tensor batch_policy;
    int row_index;
    float value;

    inline float operator[](int action_index) const {
        return batch_policy.data_ptr<float>()[row_index * batch_policy.size(1) + action_index];
    }
};
struct Inferer {
    // Inferer should have a method to predict the policy and value for a given
    // game state
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
