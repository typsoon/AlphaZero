#ifndef INFERER_H
#define INFERER_H

#include "game.hpp"
#include <torch/torch.h>

typedef std::pair<torch::Tensor, float> inference_result;
struct Inferer {
    // Inferer should have a method to predict the policy and value for a given
    // game state
    virtual std::vector<inference_result>
    infer(std::vector<GameState> gameState) = 0;
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
