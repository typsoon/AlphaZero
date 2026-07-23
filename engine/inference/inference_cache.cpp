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

// splitmix64's finalizer (Steele, Lea & Flood / Vigna) - full avalanche:
// every input bit affects every output bit. The right-shift XORs are what the
// previous FNV-style `h ^= block; h *= prime` mixing lacked: multiplication
// mod 2^64 only carries bit influence *upward*, so a difference confined to a
// block's high bytes (bytes 4-7, i.e. every ODD-indexed float of the tensor)
// could never reach the hash's low bits, and the prime's 2^40 term kept
// pushing it off the top. Measured on 100k canonical-tensor pairs differing
// only by one piece on a different odd-column square: the FNV-style block mix
// produced 370 full 64-bit collisions (0.37%!) and identical low-32-bits in
// ALL 100k pairs; this mixer (like true byte-wise FNV-1a) produced none. Those
// collisions made the cache serve a *different position's* (legal_actions,
// logits) as hits, which is exactly what sent illegal actions into MCTS
// during training - see move_piece()'s en-passant clearing in chess.cpp for
// how one shape of illegal action then corrupts the heap.
inline uint64_t mix64(uint64_t z) {
    z ^= z >> 30U;
    z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27U;
    z *= 0x94d049bb133111ebULL;
    z ^= z >> 31U;
    return z;
}

} // namespace

uint64_t InferenceCache::hash_state(const float *data, size_t count) {
    constexpr uint64_t kSeed = 0xcbf29ce484222325ULL;
    uint64_t h = kSeed;

    size_t bytes_len = count * sizeof(float);
    const auto *bytes = reinterpret_cast<const uint8_t *>(data); // NOLINT

    size_t blocks = bytes_len / 8;
    for (size_t i = 0; i < blocks; ++i) {
        uint64_t block; // NOLINT
        std::memcpy(&block, bytes + i * 8, 8);
        h = mix64(h ^ block);
    }

    if (blocks * 8 < bytes_len) {
        uint64_t tail = 0;
        std::memcpy(&tail, bytes + blocks * 8, bytes_len - blocks * 8);
        // Fold the tail's byte count in too, so buffers differing only in a
        // trailing zero byte don't alias.
        h = mix64(h ^ tail ^ (bytes_len - blocks * 8));
    }

    return h == kEmptyKey ? 1 : h;
}

bool InferenceCache::lookup(uint64_t key, inference_result &out) {
    Shard &shard = shard_for(key);
    {
        std::shared_lock<std::shared_mutex> lock(shard.mutex);
        const Entry &entry = shard.slots[slot_for(key)];
        if (entry.key == key) {
            out = entry.value;
            hit_count.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    miss_count.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void InferenceCache::insert(uint64_t key, const inference_result &value) {
    Shard &shard = shard_for(key);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    Entry &entry = shard.slots[slot_for(key)];
    entry.key = key;
    entry.value = value;
}
