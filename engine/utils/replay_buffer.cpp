#include "replay_buffer.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

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
        if (!states_buffer.defined()) {
            auto state_shape = transition.state.squeeze(0).sizes();
            std::vector<int64_t> shape{static_cast<int64_t>(capacity)};
            shape.insert(shape.end(), state_shape.begin(), state_shape.end());
            states_buffer = torch::empty(shape, transition.state.options());
            rewards_buffer = torch::empty({static_cast<int64_t>(capacity)}, torch::kFloat32);
        }

        auto slot = static_cast<int64_t>(ptr);
        states_buffer[slot].copy_(transition.state.squeeze(0));
        rewards_buffer.accessor<float, 1>()[slot] = transition.reward;

        buffer[ptr] = transition;
        // Re-point at a view into states_buffer instead of keeping the caller's
        // original tensor alive too - see states_buffer's comment in the header
        // for why this matters (avoids storing every state twice).
        buffer[ptr].state = states_buffer[slot];

        invalidate_cache_entry(ptr);
        ptr = (ptr + 1) % capacity;
        if (size < capacity) {
            size++;
        }
        total_added++;
    }
}

size_t ReplayBuffer::get_size() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex);
    return size;
}

size_t ReplayBuffer::get_total_added() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex);
    return total_added;
}

void ReplayBuffer::save(const std::string &path) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex);
    auto n = static_cast<int64_t>(size);

    torch::Tensor states;
    torch::Tensor rewards;
    torch::Tensor idx_all;
    torch::Tensor val_all;
    torch::Tensor lengths;
    if (n == 0 || !states_buffer.defined()) {
        states = torch::empty({0});
        rewards = torch::empty({0}, torch::kFloat32);
        idx_all = torch::empty({0}, torch::kInt64);
        val_all = torch::empty({0}, torch::kFloat32);
        lengths = torch::empty({0}, torch::kInt64);
    } else {
        // Valid entries always occupy slots [0, size): while size < capacity the
        // ring hasn't wrapped, and once full n == capacity covers every slot.
        // clone() detaches from the live buffers so concurrent add()s can't
        // mutate what we're serializing after the lock is released.
        states = states_buffer.narrow(0, 0, n).clone();
        rewards = rewards_buffer.narrow(0, 0, n).clone();
        std::vector<torch::Tensor> idx_parts;
        std::vector<torch::Tensor> val_parts;
        idx_parts.reserve(n);
        val_parts.reserve(n);
        std::vector<int64_t> lens(n);
        for (int64_t i = 0; i < n; ++i) {
            const Transition &t = buffer[i];
            idx_parts.push_back(t.policy_indices);
            val_parts.push_back(t.policy_values);
            lens[i] = t.policy_indices.numel();
        }
        idx_all = idx_parts.empty() ? torch::empty({0}, torch::kInt64)
                                    : torch::cat(idx_parts).to(torch::kInt64).clone();
        val_all = val_parts.empty() ? torch::empty({0}, torch::kFloat32)
                                    : torch::cat(val_parts).to(torch::kFloat32).clone();
        lengths = torch::tensor(lens, torch::kInt64);
    }
    // Trailing metadata tensor lets a reader defensively cross-check even
    // without the Python sidecar: {action_size, num_transitions}.
    torch::Tensor meta = torch::tensor({action_size, n}, torch::kInt64);
    std::vector<torch::Tensor> archive{states, rewards, idx_all, val_all, lengths, meta};
    torch::save(archive, path);
}

void ReplayBuffer::load(const std::string &path) {
    std::vector<torch::Tensor> archive;
    torch::load(archive, path);
    if (archive.size() < 5) {
        throw std::runtime_error("ReplayBuffer::load: malformed archive (expected >=5 "
                                 "tensors) in '" +
                                 path + "'");
    }
    const torch::Tensor &states = archive[0];
    const torch::Tensor &rewards = archive[1];
    const torch::Tensor &idx_all = archive[2];
    const torch::Tensor &val_all = archive[3];
    const torch::Tensor &lengths = archive[4];

    auto n = lengths.numel();
    if (n == 0) {
        return;
    }
    if (archive.size() >= 6) {
        auto loaded_action_size = archive[5].accessor<int64_t, 1>()[0];
        if (loaded_action_size != action_size) {
            throw std::runtime_error("ReplayBuffer::load: action_size mismatch (archive " +
                                     std::to_string(loaded_action_size) + " vs buffer " +
                                     std::to_string(action_size) + ") in '" + path +
                                     "'; the saved policy targets index a different action space");
        }
    }

    auto rewards_a = rewards.accessor<float, 1>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    std::vector<Transition> transitions;
    transitions.reserve(n);
    int64_t off = 0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t len = lengths_a[i];
        // narrow() bounds-checks: a corrupt lengths/idx mismatch throws here.
        transitions.emplace_back(states[i], idx_all.narrow(0, off, len),
                                 val_all.narrow(0, off, len), rewards_a[i]);
        off += len;
    }
    add(transitions);
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

    if (batch_size == 0) {
        return {torch::empty({0}), torch::empty({0}), torch::empty({0})};
    }

    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Sampled rows are written directly into pinned-memory destination tensors
    // below, rather than collected into loose per-row tensors and
    // torch::stack()-ed into a freshly allocated, regular (pageable) tensor -
    // so that the caller's later .to(device, non_blocking=True) is a genuine
    // async, high-bandwidth H2D transfer instead of silently falling back to a
    // blocking copy (which is what a pageable source forces regardless of
    // non_blocking=True). Torch's caching host allocator pools pinned blocks
    // internally and only recycles one once it can prove (via CUDA event
    // tracking) that no in-flight transfer still reads it, so allocating a
    // "fresh" pinned tensor on every call here is both safe and, after
    // warmup, cheap - unlike a hand-rolled persistent pinned buffer, which
    // would risk the next call's writes racing an still-in-flight transfer
    // from the previous one.
    bool pin = torch::cuda::is_available();
    auto state_shape = states_buffer.sizes().slice(1);
    std::vector<int64_t> states_shape{static_cast<int64_t>(batch_size)};
    states_shape.insert(states_shape.end(), state_shape.begin(), state_shape.end());

    torch::Tensor states = torch::empty(states_shape, states_buffer.options().pinned_memory(pin));
    torch::Tensor policies =
        torch::empty({static_cast<int64_t>(batch_size), action_size},
                     torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(pin));
    torch::Tensor rewards =
        torch::empty({static_cast<int64_t>(batch_size)},
                     torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(pin));

    // states and rewards are pulled out of their [capacity, ...] buffers with one
    // gather call each, rather than one .copy_()/accessor write per sampled row -
    // both this batch's rows and, for states, the source data itself are already
    // fully materialized (unlike policies, whose dense form may still need
    // computing below), so there's nothing per-row left to do for these two: the
    // whole minibatch's worth of data movement happens in a single vectorized,
    // internally-parallelized ATen call instead of batch_size separate ones, each
    // of which pays its own fixed dispatcher overhead regardless of how little
    // data it moves.
    std::vector<int64_t> indices_i64(indices.begin(),
                                     indices.begin() + static_cast<int64_t>(batch_size));
    torch::Tensor index_tensor =
        torch::from_blob(indices_i64.data(), {static_cast<int64_t>(batch_size)},
                         torch::TensorOptions().dtype(torch::kInt64))
            .clone(); // clone: indices_i64 is a local vector, freed at function return
    torch::index_select_out(states, states_buffer, /*dim=*/0, index_tensor);
    torch::index_select_out(rewards, rewards_buffer, /*dim=*/0, index_tensor);

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

        policies[static_cast<int64_t>(i)].copy_(*cached);
    }

    return {states, policies, rewards};
}
