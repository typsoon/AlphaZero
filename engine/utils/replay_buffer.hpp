#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include <list>
#include <optional>
#include <random>
#include <shared_mutex>
#include <torch/torch.h>
#include <tuple>
#include <vector>

struct Transition {
    torch::Tensor state;
    // Sparse policy target: MCTS visits only ~20-40 of the thousands of encoded
    // actions per position (e.g. 20480 for chess), so a dense vector is >90%
    // zeros and dominates the buffer's memory footprint. Only the nonzero
    // (index, value) pairs are kept; ReplayBuffer expands them back to a dense
    // row on demand at sample() time - see dense_policy_cache below for why
    // that expansion is cached instead of redone on every draw.
    torch::Tensor policy_indices; // int64
    torch::Tensor policy_values;  // float32, same length as policy_indices
    float reward;

    Transition(torch::Tensor s = {}, torch::Tensor idx = {}, torch::Tensor val = {}, float r = 0);
};

class ReplayBuffer {
    std::vector<Transition> buffer;
    size_t ptr = 0, size = 0;
    size_t capacity;
    int64_t action_size;
    mutable std::shared_mutex rw_mutex;
    mutable std::mt19937 rng;

    // Dense, contiguous [capacity, ...state_shape] / [capacity] mirrors of every
    // transition's state and reward, kept alongside `buffer` so sample() can pull
    // a whole minibatch out via a single torch::index_select_out() call instead of
    // one .copy_()/accessor write per sampled row - see sample()'s comment for
    // why that matters. Lazily shaped on the first add() call, once a state
    // tensor's shape is known. Unlike dense_policy_cache, this duplicates nothing:
    // every transition's state already has to be stored somewhere regardless, and
    // add() re-points buffer[ptr].state at a view into states_buffer instead of
    // keeping the caller's original tensor around, so there's exactly one copy of
    // each state in memory, not two.
    torch::Tensor states_buffer;
    torch::Tensor rewards_buffer;

    // Bounds dense_policy_cache below regardless of capacity - see that
    // member for why an unbounded (capacity-sized) cache is dangerous.
    size_t max_cache_entries;

    // Densified policy rows, one slot per buffer index. Populated lazily the
    // first time a slot is sampled and reused by every later sample() call
    // that draws the same slot again, up to max_cache_entries total - beyond
    // that, the least-recently-used entry is evicted (see lru_order/
    // lru_position below) to make room. Without a bound tied to
    // max_cache_entries rather than capacity, this can grow to capacity *
    // ~82KB/entry for chess (26GB at a 320,000-entry buffer), which directly
    // contributed to OOM kills during training. Invalidated per-slot in add()
    // (the old cached row would otherwise describe stale data once that slot
    // is overwritten) and dropped entirely by clear_dense_cache().
    mutable std::vector<std::optional<torch::Tensor>> dense_policy_cache;

    // LRU bookkeeping for dense_policy_cache: lru_order holds buffer indices,
    // most-recently-used at the front; lru_position[i] is that index's
    // iterator into lru_order (if cached), for O(1) removal/promotion instead
    // of a linear scan on every access.
    mutable std::list<size_t> lru_order;
    mutable std::vector<std::optional<std::list<size_t>::iterator>> lru_position;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample(size_t batch_size) const;

    void clear_dense_cache() const;

    // Drops slot idx's cache entry (if any) and its LRU bookkeeping. Used both
    // by add() (the slot's data just changed under it) and by sample()'s
    // eviction (making room for a new entry once max_cache_entries is hit).
    void invalidate_cache_entry(size_t idx) const;

    // Marks slot idx most-recently-used, inserting it into lru_order/
    // lru_position if it isn't already tracked.
    void touch_lru(size_t idx) const;

  public:
    // max_cache_entries bounds dense_policy_cache independently of capacity -
    // pass how large you actually want the sparse-to-dense cache to grow
    // (e.g. one minibatch's worth), not the full replay buffer size. Default
    // is a fixed, capacity-independent size (not e.g. a fraction of capacity)
    // so it stays safe even if capacity is scaled up later without a caller
    // thinking to revisit this value.
    ReplayBuffer(size_t capacity, int64_t action_size, size_t max_cache_entries = 4096);

    void add(const std::vector<Transition> &transitions);

    size_t get_size() const;

    // Persist the buffer's live transitions to `path` as a torch archive so a
    // later run can reload them instead of refilling from empty self-play (the
    // ~15-20 min ramp seen on every restart). Only the `size` valid entries are
    // written; ring-buffer order is irrelevant since sampling is uniform. The
    // sparse policy targets are packed CSR-style (one concatenated index/value
    // tensor plus a per-transition length tensor) to avoid a variable-length
    // list. The COMPATIBILITY GUARD (encoder tag / action_size / state shape)
    // lives in the Python caller: this method blindly serializes whatever is in
    // the buffer, and load() blindly trusts the file, so callers MUST validate
    // metadata before load()-ing (see python/__main__.py). action_size and the
    // state shape are recorded here too, so a defensive check is possible.
    void save(const std::string &path) const;

    // Reconstruct transitions from a save() archive and add() them. Appends to
    // whatever is already buffered (normally called once on a fresh buffer).
    // No shape/action validation - the caller owns that (see save()).
    void load(const std::string &path);

    // RAII handle onto one span of repeated sampling. Reusing one handle
    // across many sample() calls is fine memory-wise regardless of how many
    // - dense_policy_cache is LRU-bounded at max_cache_entries independently
    // of how the handle is used. sample() is only reachable through this
    // handle - there is no way to populate dense_policy_cache without also
    // getting a guarantee that it will be freed again. The moment the handle
    // is destroyed (falls out of scope in Python, or is explicitly deleted),
    // it calls clear_dense_cache() on the buffer for you, with no explicit
    // cleanup call needed.
    class CachedSampler {
        const ReplayBuffer *buffer;

      public:
        explicit CachedSampler(const ReplayBuffer *buffer) : buffer(buffer) {}
        ~CachedSampler() { close(); }

        CachedSampler(const CachedSampler &) = delete;
        CachedSampler &operator=(const CachedSampler &) = delete;
        CachedSampler(CachedSampler &&other) noexcept : buffer(other.buffer) {
            other.buffer = nullptr;
        }
        CachedSampler &operator=(CachedSampler &&other) noexcept {
            if (this != &other) {
                close();
                buffer = other.buffer;
                other.buffer = nullptr;
            }
            return *this;
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample(size_t batch_size) const {
            return buffer->sample(batch_size);
        }

        // Triggers clear_dense_cache() early. Exposed (bound as __exit__ in
        // bindings.cpp) so `with buf.get_sampler() as s:` in Python
        // guarantees cleanup right at the block boundary, rather than
        // whenever CPython's refcounting happens to drop the last reference -
        // usually the same moment, but not if something else were holding on
        // to the handle too. clear_dense_cache() is idempotent, so calling
        // close() more than once (e.g. via __exit__ and then the destructor)
        // or sample()-ing again afterward is harmless, just redundant/wasted
        // work - buffer is only ever null here for a moved-from handle.
        void close() {
            if (buffer != nullptr) {
                buffer->clear_dense_cache();
            }
        }
    };

    CachedSampler get_sampler() const { return CachedSampler(this); }
};

#endif // REPLAY_BUFFER_HPP
