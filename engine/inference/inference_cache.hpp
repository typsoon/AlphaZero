#ifndef INFERENCE_CACHE_HPP
#define INFERENCE_CACHE_HPP

#include "inferer.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <vector>

// Transposition cache for NN inference results, shared by every self-play
// thread of one NetworkInfererFactory (and therefore scoped to one network's
// weights - it must never outlive them, since a cached (policy, value) pair is
// only valid for the exact network that produced it; the factory owning both
// the network and the cache enforces that by construction).
//
// Keying: callers hash the *canonical state tensor's content* (hash_state()
// over the exact float buffer that would be fed to the network), not any
// game-specific position hash. That's deliberate: identical key <=> identical
// network input <=> identical (deterministic) network output, for any game.
// Chess's Zobrist hash specifically is NOT a safe key here - it excludes
// move_count, which Chess::write_canonical_state() writes into one of the
// input planes, so two Zobrist-equal positions can produce different network
// inputs and the cache would silently serve wrong results for one of them.
//
// Structure: N independently-locked shards, each a fixed-size open-addressed
// slot array with always-replace eviction (the classic chess-engine
// transposition-table scheme). Bounded by construction - max_entries slots
// are allocated up front and a colliding insert overwrites, so memory can't
// grow with the (effectively unbounded) number of distinct positions a long
// self-play run encounters; this codebase has an OOM-kill history from a
// cache that scaled with its workload instead of a hard cap (see
// ReplayBuffer::dense_policy_cache). Always-replace also keeps reads
// compatible with std::shared_mutex's shared mode: unlike LRU, a lookup
// doesn't mutate any recency bookkeeping, so concurrent readers of one shard
// never contend with each other - only with the occasional writer.
//
// A 64-bit key collision (two different tensors hashing equal) would serve a
// wrong result; at ~10^7 distinct positions per training iteration the
// birthday-bound probability is ~10^-5 per iteration, and a single wrong
// (policy, value) among millions of MCTS evaluations is noise - accepted, as
// every engine transposition table does.
class InferenceCache {
  public:
    // max_entries == 0 is a valid "disabled" configuration at the call sites
    // (they just don't construct an InferenceCache); this class itself
    // requires max_entries > 0. num_shards is rounded up to a power of two.
    explicit InferenceCache(size_t max_entries, size_t num_shards = 16);

    // Fast non-cryptographic hash over the tensor bytes. Never returns
    // kEmptyKey, so the raw hash is always a valid occupied-slot marker.
    static uint64_t hash_state(const float *data, size_t count);

    // Returns the cached shared_ptr on a hit (zero-copy — no vector allocation),
    // or nullptr on a miss. Shared (reader) lock on one shard.
    std::shared_ptr<const inference_result> lookup(uint64_t key);

    // Overwrites whatever occupied the slot (always-replace). Exclusive
    // (writer) lock on one shard. Takes a shared_ptr so the same heap object
    // can be inserted into the cache without an extra copy.
    void insert(uint64_t key, std::shared_ptr<const inference_result> value);

    uint64_t hits() const { return hit_count.load(std::memory_order_relaxed); }
    uint64_t misses() const { return miss_count.load(std::memory_order_relaxed); }

  private:
    static constexpr uint64_t kEmptyKey = 0;

    struct Entry {
        uint64_t key = kEmptyKey;
        std::shared_ptr<const inference_result> value; // null == empty
    };

    // Heap-allocated per shard because std::shared_mutex is neither movable
    // nor copyable, which a plain std::vector<Shard> would require.
    struct Shard {
        std::shared_mutex mutex;
        std::vector<Entry> slots;
    };

    // Shard from the high bits, slot from the low bits, so the two indices
    // don't correlate (using low bits for both would map each shard's slots
    // from only 1/num_shards of the slot space).
    Shard &shard_for(uint64_t key) { return *shards[(key >> 48U) & (shards.size() - 1)]; }
    size_t slot_for(uint64_t key) const { return key % slots_per_shard; }

    size_t slots_per_shard;
    std::vector<std::unique_ptr<Shard>> shards;
    std::atomic<uint64_t> hit_count{0};
    std::atomic<uint64_t> miss_count{0};
};

#endif // INFERENCE_CACHE_HPP
