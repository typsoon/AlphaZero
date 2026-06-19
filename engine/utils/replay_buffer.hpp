#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include <mutex>
#include <random>
#include <torch/torch.h>
#include <tuple>
#include <vector>

struct Transition {
    torch::Tensor state;
    torch::Tensor policy;
    float reward;

    Transition(const torch::Tensor &s = {}, const torch::Tensor &p = {}, float r = 0);
};

class ReplayBuffer {
    std::vector<Transition> buffer;
    size_t ptr = 0, size = 0;
    size_t capacity;
    std::mutex write_mutex;
    mutable std::mt19937 rng;

  public:
    explicit ReplayBuffer(size_t capacity);

    void add(const std::vector<Transition> &transitions);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample(size_t batch_size) const;
};

#endif // REPLAY_BUFFER_HPP
