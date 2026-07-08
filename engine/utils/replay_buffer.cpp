#include "replay_buffer.hpp"
#include <algorithm>
#include <numeric>

Transition::Transition(torch::Tensor s, torch::Tensor idx, torch::Tensor val, float r)
    : state(std::move(s)), policy_indices(std::move(idx)), policy_values(std::move(val)),
      reward(r) {}

ReplayBuffer::ReplayBuffer(size_t capacity, int64_t action_size, size_t max_cache_entries) // NOLINT
    : capacity(capacity), action_size(action_size), max_cache_entries(max_cache_entries),
      buffer(capacity), rng(std::random_device{}()), dense_policy_cache(capacity),
      lru_position(capacity) {}

void ReplayBuffer::invalidate_cache_entry(size_t idx) const {
    auto &pos = lru_position[idx];
    if (pos.has_value()) {
        lru_order.erase(*pos);
        pos.reset();
    }
    dense_policy_cache[idx].reset();
}

void ReplayBuffer::touch_lru(size_t idx) const {
    auto &pos = lru_position[idx];
    if (pos.has_value()) {
        lru_order.erase(*pos);
    }
    lru_order.push_front(idx);
    pos = lru_order.begin();
}

void ReplayBuffer::add(const std::vector<Transition> &transitions) { // NOLINT
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    for (const auto &transition : transitions) {
        buffer[ptr] = transition;
        invalidate_cache_entry(ptr);
        ptr = (ptr + 1) % capacity;
        if (size < capacity) {
            size++;
        }
    }
}

size_t ReplayBuffer::get_size() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex);
    return size;
}

void ReplayBuffer::clear_dense_cache() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    std::fill(dense_policy_cache.begin(), dense_policy_cache.end(), std::nullopt);
    std::fill(lru_position.begin(), lru_position.end(), std::nullopt);
    lru_order.clear();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::sample(size_t batch_size) const {
    // Unique (not shared) lock: this may populate dense_policy_cache entries
    // below, and add() invalidates entries under the same lock. In the current
    // usage pattern sample() is only ever called sequentially from the
    // training loop (never concurrently with itself), so this doesn't cost any
    // real concurrency - it just keeps the lazy-cache-fill logic race-free.
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    batch_size = std::min(batch_size, size);

    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> policies;
    std::vector<float> rewards;

    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = indices[i];
        const Transition &transition = buffer[idx];
        auto &cached = dense_policy_cache[idx];

        if (cached.has_value()) {
            touch_lru(idx);
        } else {
            if (max_cache_entries > 0 && lru_order.size() >= max_cache_entries) {
                invalidate_cache_entry(lru_order.back());
            }
            torch::Tensor dense = torch::zeros({action_size}, torch::kFloat32);
            dense.scatter_(0, transition.policy_indices, transition.policy_values);
            cached = dense;
            touch_lru(idx);
        }

        states.push_back(transition.state.squeeze(0));
        policies.push_back(*cached);
        rewards.push_back(transition.reward);
    }

    return {torch::stack(states), torch::stack(policies), torch::tensor(rewards, torch::kFloat32)};
}
