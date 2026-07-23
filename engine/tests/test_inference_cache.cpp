#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/MemoryLeakWarningPlugin.h"
#include "CppUTest/TestHarness.h"
#include "inference/inference_cache.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

inference_result make_result(int seed) {
    inference_result r;
    r.legal_actions = {seed, seed + 1, seed + 2};
    r.legal_action_logits = {static_cast<float>(seed), static_cast<float>(seed) * 0.5f, -1.0f};
    r.value = static_cast<float>(seed) / 100.0f;
    return r;
}

void check_result_equal(const inference_result &expected, const inference_result &actual) {
    CHECK_EQUAL(expected.legal_actions.size(), actual.legal_actions.size());
    for (size_t i = 0; i < expected.legal_actions.size(); ++i) {
        CHECK_EQUAL(expected.legal_actions[i], actual.legal_actions[i]);
        DOUBLES_EQUAL(expected.legal_action_logits[i], actual.legal_action_logits[i], 1e-9);
    }
    DOUBLES_EQUAL(expected.value, actual.value, 1e-9);
}

} // namespace

TEST_GROUP(InferenceCacheTests){};

TEST(InferenceCacheTests, InsertThenLookupRoundTrips) {
    InferenceCache cache(/*max_entries=*/1024);
    auto res = make_result(7);
    cache.insert(42, std::make_shared<const inference_result>(res));

    auto out = cache.lookup(42);
    CHECK_TRUE(out != nullptr);
    check_result_equal(res, *out);
    CHECK_EQUAL(1, static_cast<int>(cache.hits()));
}

TEST(InferenceCacheTests, MissingKeyIsAMiss) {
    InferenceCache cache(/*max_entries=*/1024);
    auto out = cache.lookup(42);
    CHECK_TRUE(out == nullptr);
    CHECK_EQUAL(1, static_cast<int>(cache.misses()));
    CHECK_EQUAL(0, static_cast<int>(cache.hits()));
}

// The bounding property this cache exists for (see the OOM history referenced
// in inference_cache.hpp): inserting far more distinct keys than max_entries
// must not grow memory - colliding keys overwrite (always-replace), and the
// evicted key becomes a miss while the evicting key is servable. With
// max_entries slots across shards, two keys that agree on shard bits and
// slot index share a slot; constructing such a pair directly (rather than
// spraying random keys) makes the eviction deterministic to assert on.
TEST(InferenceCacheTests, CollidingKeyEvictsPreviousEntry) {
    // 16 shards (default) x 1 slot each: any two keys with equal shard bits
    // (bits 48+) collide. Keys 1 and 2 both have shard bits 0.
    InferenceCache cache(/*max_entries=*/16);
    cache.insert(1, std::make_shared<const inference_result>(make_result(1)));
    cache.insert(2, std::make_shared<const inference_result>(make_result(2)));

    CHECK_TRUE(cache.lookup(1) == nullptr); // evicted by key 2
    auto out = cache.lookup(2);
    CHECK_TRUE(out != nullptr);
    check_result_equal(make_result(2), *out);
}

TEST(InferenceCacheTests, HashStateDiffersOnAnySingleValueChange) {
    std::vector<float> a(static_cast<size_t>(19 * 8 * 8), 0.0f);
    auto b = a;
    b[static_cast<size_t>(13 * 64)] = 1.0f; // e.g. chess's move_count plane - the
                                            // Zobrist blind spot

    CHECK_TRUE(InferenceCache::hash_state(a.data(), a.size()) !=
               InferenceCache::hash_state(b.data(), b.size()));
    // Deterministic: same content, same hash.
    CHECK_TRUE(InferenceCache::hash_state(a.data(), a.size()) ==
               InferenceCache::hash_state(a.data(), a.size()));
}

// Regression test for a real cache-poisoning bug hit during chess training:
// hash_state used to mix 8-byte blocks FNV-style (h ^= block; h *= prime),
// which has no downward avalanche - an odd-indexed float only occupies bytes
// 4-7 of its block, so its value could only ever influence bits 32-63 of the
// hash, and multiplication mod 2^64 carries upward only. Two positions
// differing solely in odd-column piece placement (e.g. "1.b3 X" vs "1.f3 X")
// therefore had IDENTICAL low hash bits always, and collided on the full 64
// bits ~0.4% of the time - at millions of lookups per training iteration the
// cache constantly served one position's (legal_actions, logits) as a hit for
// the other, sending illegal actions into MCTS (and, through move_piece()'s
// en-passant clearing, corrupting the heap). This exhaustively checks the
// exact colliding family: a single 1.0 moved between two odd cells must
// always change the full hash, and must change the LOW 32 bits in the vast
// majority of pairs (the broken mixer changed them in exactly zero).
TEST(InferenceCacheTests, HashStateAvalanchesOddIndexedCellChanges) {
    constexpr size_t kStateSize = 19 * 8 * 8ULL;
    std::vector<float> base(kStateSize, 0.0f);
    for (size_t i = 12 * 64ULL; i < kStateSize; ++i)
        base[i] = 1.0f; // constant planes, as in a real canonical state

    int total = 0;
    int full_collisions = 0;
    int low32_matches = 0;
    for (int plane = 0; plane < 12; ++plane) {
        for (size_t from = 1; from < 64; from += 2) {
            for (size_t to = 1; to < 64; to += 2) {
                if (from == to)
                    continue;
                auto a = base;
                a[plane * 64ULL + from] = 1.0f;
                auto b = base;
                b[plane * 64ULL + to] = 1.0f;
                uint64_t ka = InferenceCache::hash_state(a.data(), a.size());
                uint64_t kb = InferenceCache::hash_state(b.data(), b.size());
                ++total;
                if (ka == kb)
                    ++full_collisions;
                if (static_cast<uint32_t>(ka) == static_cast<uint32_t>(kb))
                    ++low32_matches;
            }
        }
    }
    CHECK_EQUAL(0, full_collisions);
    // A sound 64-bit hash matches any fixed 32 bits with p ~= 2^-32; allow a
    // token margin. The broken mixer scored low32_matches == total here.
    CHECK_TRUE(low32_matches < total / 100);
}

// Concurrency smoke test: readers and writers hammer overlapping keys across
// every shard; TSan/ASan builds would flag races, and even without them a
// torn read would produce a result whose contents don't match its key's
// seeded values. Keys are striped so each key always maps to one specific
// seeded result, making corruption detectable.
TEST(InferenceCacheTests, ConcurrentReadersAndWritersStayConsistent) {
    InferenceCache cache(/*max_entries=*/4096);
    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 20000;
    constexpr int kKeySpace = 512;
    std::atomic<bool> corruption{false};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                // Spread keys across the full 64-bit space so all shards get
                // traffic (shard index comes from bits 48+).
                uint64_t small = static_cast<uint64_t>((i * 31 + t * 7) % kKeySpace) + 1;
                uint64_t key = small | (small << 48U);
                if ((i + t) % 3 == 0) {
                    cache.insert(key, std::make_shared<const inference_result>(
                                          make_result(static_cast<int>(small))));
                } else {
                    auto out = cache.lookup(key);
                    if (out) {
                        if (out->legal_actions.size() != 3 ||
                            out->legal_actions[0] != static_cast<int>(small)) {
                            corruption = true;
                        }
                    }
                }
            }
        });
    }
    for (auto &th : threads)
        th.join();

    CHECK_FALSE(corruption.load());
    CHECK_TRUE(cache.hits() + cache.misses() > 0);
}

int main(int ac, char **av) {
    MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
    return CommandLineTestRunner::RunAllTests(ac, av);
}
