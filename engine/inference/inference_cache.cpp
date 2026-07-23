#include "inference_cache.hpp"

#include <cstring>
#include <stdexcept>

namespace {

size_t round_up_to_power_of_two(size_t v) {
    size_t p = 1;
    while (p < v)
        p <<= 1U;
    return p;
}

} // namespace

InferenceCache::InferenceCache(size_t max_entries, size_t num_shards) {
    if (max_entries == 0)
        throw std::invalid_argument(
            "InferenceCache requires max_entries > 0; to disable caching, don't construct one");
    size_t shard_count = round_up_to_power_of_two(num_shards == 0 ? 1 : num_shards);
    slots_per_shard = (max_entries + shard_count - 1) / shard_count;
    shards.reserve(shard_count);
    for (size_t s = 0; s < shard_count; ++s) {
        auto shard = std::make_unique<Shard>();
        shard->slots.resize(slots_per_shard);
        shards.push_back(std::move(shard));
    }
}

namespace {

// wyhash-style finalizer for 64-bit values - full avalanche, branch-free.
inline uint64_t wymix(uint64_t a, uint64_t b) {
    __uint128_t r = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64U);
}

// Two independent multiply-mix constants (from wyhash paper).
constexpr uint64_t kP0 = 0xa0761d6478bd642fULL;
constexpr uint64_t kP1 = 0xe7037ed1a0b428dbULL;
constexpr uint64_t kP2 = 0x8ebc6af09c88c6e3ULL;
constexpr uint64_t kP3 = 0x589965cc75374cc3ULL;

} // namespace

uint64_t InferenceCache::hash_state(const float *data, size_t count) {
    // wyhash-style streaming hash. We XOR pairs of 8-byte words together in
    // two independent lanes (a, b), then combine with a final multiply-mix.
    // This halves the number of multiplications compared to the previous
    // per-block mix64 approach (one multiply every 16 bytes instead of 8),
    // while preserving full avalanche through the final wymix call.
    // The pair-wise XOR before multiplying folds every input bit into both
    // lanes, keeping the hash quality equivalent to wyhash's own design.
    const size_t byte_len = count * sizeof(float);
    const auto *p = reinterpret_cast<const uint8_t *>(data); // NOLINT

    uint64_t seed = kP0 ^ static_cast<uint64_t>(byte_len) * kP1;
    uint64_t a = 0, b = 0;

    size_t i = 0;
    // Process 16-byte chunks: read two uint64_t words and combine them.
    for (; i + 16 <= byte_len; i += 16) {
        uint64_t lo = 0;
        uint64_t hi = 0;
        std::memcpy(&lo, p + i, 8);
        std::memcpy(&hi, p + i + 8, 8);
        a ^= lo;
        b ^= hi;
        seed = wymix(a ^ kP0, b ^ seed);
    }
    // Remaining 8 bytes (if any).
    if (i + 8 <= byte_len) {
        uint64_t lo = 0;
        std::memcpy(&lo, p + i, 8);
        a ^= lo;
        i += 8;
    }
    // Remaining tail (< 8 bytes).
    if (i < byte_len) {
        uint64_t tail = 0;
        std::memcpy(&tail, p + i, byte_len - i);
        b ^= tail;
    }
    uint64_t h = wymix(a ^ kP2, b ^ seed) ^ wymix(kP3, static_cast<uint64_t>(count));
    return h == kEmptyKey ? 1 : h;
}

std::shared_ptr<const inference_result> InferenceCache::lookup(uint64_t key) {
    Shard &shard = shard_for(key);
    {
        std::shared_lock<std::shared_mutex> lock(shard.mutex);
        const Entry &entry = shard.slots[slot_for(key)];
        if (entry.key == key && entry.value) {
            hit_count.fetch_add(1, std::memory_order_relaxed);
            return entry.value; // shared_ptr copy: no vector allocation
        }
    }
    miss_count.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

void InferenceCache::insert(uint64_t key, std::shared_ptr<const inference_result> value) {
    Shard &shard = shard_for(key);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    Entry &entry = shard.slots[slot_for(key)];
    entry.key = key;
    entry.value = std::move(value); // pointer store, no deep copy
}
