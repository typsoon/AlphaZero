#ifndef ALPHA_ZERO_TRAINER_HPP
#define ALPHA_ZERO_TRAINER_HPP

#include "network.hpp"
#include "replay_buffer.hpp"
#include <torch/torch.h>

class AlphaZeroTrainer {
    AlphaZeroNetwork &network;
    ReplayBuffer &replay_buffer;
    torch::optim::Optimizer &optimizer;
    torch::Device device;
    size_t minibatch_size;

  public:
    AlphaZeroTrainer(AlphaZeroNetwork &network, ReplayBuffer &replay_buffer,
                     torch::optim::Optimizer &optimizer, torch::Device device = torch::kCUDA,
                     size_t minibatch_size = 4096);

    void train(size_t train_steps = 1000, size_t batch_size = 64);
};

#endif // !ALPHA_ZERO_TRAINER_HPP
