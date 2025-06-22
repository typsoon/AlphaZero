#ifndef MCTS_HPP
#define MCTS_HPP

#include "game.hpp"
#include "inferer.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>

using std::make_unique;
using std::unique_ptr;
using std::vector;

class MCTS {
    using InfererPtr = unique_ptr<Inferer>;
    InfererPtr network;
    float c_init;
    float c_base;
    float eps;
    float alpha;
    torch::Device device;

  public:
    MCTS(unique_ptr<Inferer> &&network, float c_init = 1.25f,
         float c_base = 19652.0f, float eps = 0.25f, float alpha = 0.3f);

    MCTS(std::string network_path, torch::Device device, float c_init = 1.25f,
         float c_base = 19652.0f, float eps = 0.25f, float alpha = 0.3f);

    std::vector<float> search(const Game &game, int num_simulations = 800,
                              int batch_size = 32);

  private:
    class Node;
    void evaluate_batch(std::vector<Node *> &leaves);

    std::vector<float>
    get_policy_from_logits(torch::Tensor policy_logits,
                           const std::vector<int> &legal_actions,
                           bool dirichletNoise = false);

    static std::vector<float> sample_dirichlet(const std::vector<float> &alpha);
};

#endif // MCTS_HPP
