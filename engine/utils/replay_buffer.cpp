#include "replay_buffer.hpp"
#include <algorithm>
#include <numeric>

Transition::Transition(const torch::Tensor &s, const torch::Tensor &p, float r)
    : state(s), policy(p), reward(r) {}

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity(capacity), buffer(capacity), rng(std::random_device{}()) {}

void ReplayBuffer::add(const std::vector<Transition> &transitions) {
    std::lock_guard<std::mutex> lock(write_mutex);
    for (const auto &transition : transitions) {
        buffer[ptr] = transition;
        ptr = (ptr + 1) % capacity;
        if (size < capacity) {
            size++;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::sample(size_t batch_size) const {
    batch_size = std::min(batch_size, size);

    std::vector<torch::Tensor> states, policies;
    std::vector<float> rewards;

    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = indices[i];
        const Transition &transition = buffer[idx];
        states.push_back(transition.state.squeeze(0));
        policies.push_back(transition.policy);
        rewards.push_back(transition.reward);
    }

    return {torch::stack(states), torch::stack(policies), torch::tensor(rewards, torch::kFloat32)};
}
