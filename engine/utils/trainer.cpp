#include "trainer.hpp"
#include "network.hpp"
#include "replay_buffer.hpp"
#include <cassert>
#include <iostream>
#include <torch/torch.h>

AlphaZeroTrainer::AlphaZeroTrainer(AlphaZeroNetwork &network, ReplayBuffer &replay_buffer,
                                   torch::optim::Optimizer &optimizer, torch::Device device,
                                   size_t minibatch_size)
    : network(network), replay_buffer(replay_buffer), optimizer(optimizer), device(device),
      minibatch_size(minibatch_size) {}

void AlphaZeroTrainer::train(size_t train_steps, size_t batch_size) {
    network->train();
    const size_t accum_steps = minibatch_size / batch_size;
    assert(minibatch_size % batch_size == 0 && accum_steps > 0);

    for (size_t step = 0; step < train_steps; step++) {
        auto [states, policies, rewards] = replay_buffer.sample(minibatch_size);
        states = states.to(device);
        policies = policies.to(device);
        rewards = rewards.to(device);

        optimizer.zero_grad();

        torch::Tensor policy_loss;
        torch::Tensor value_loss;

        for (size_t i = 0; i < accum_steps; i++) {
            size_t start = i * batch_size;
            size_t end = start + batch_size;

            auto state_batch = states.narrow(0, start, batch_size);
            auto pi_batch = policies.narrow(0, start, batch_size);
            auto reward_batch = rewards.narrow(0, start, batch_size);

            auto [policy_logits, value] = network->forward(state_batch);

            auto logp = torch::log_softmax(policy_logits, 1);
            policy_loss = -torch::sum(logp * pi_batch, 1).mean();
            value_loss = torch::mse_loss(value.squeeze(), reward_batch);
            auto loss = policy_loss + value_loss;
            loss = loss / static_cast<double>(accum_steps);
            loss.backward();
        }

        optimizer.step();

#ifndef NDEBUG
        std::cout << "Step: " << step << ", Policy Loss: " << policy_loss.item<float>()
                  << ", Value Loss: " << value_loss.item<float>() << std::endl;
#endif
    }
}
