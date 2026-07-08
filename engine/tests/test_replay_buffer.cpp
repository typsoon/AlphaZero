#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/MemoryLeakWarningPlugin.h"
#include "CppUTest/TestHarness.h"
#include "replay_buffer.hpp"
#include <spdlog/spdlog.h>
#include <torch/torch.h>

namespace {

constexpr int64_t kActionSize = 6;

// Builds a Transition whose state is filled with a single distinguishable value
// and whose sparse policy has been expanded into (indices, values) - mirrors how
// self_play.cpp's sparsify_policy() builds the tensors that actually get stored.
// The state carries an extra leading dim (size 1) since ReplayBuffer::sample()
// calls state.squeeze(0), matching how self_play.cpp stores
// game_state_tensor.unsqueeze(0).
Transition make_transition(float state_fill, const std::vector<int64_t> &indices,
                           const std::vector<float> &values, float reward) {
    torch::Tensor state = torch::full({1, 3}, state_fill, torch::kFloat32);
    torch::Tensor idx = torch::tensor(indices, torch::kInt64);
    torch::Tensor val = torch::tensor(values, torch::kFloat32);
    return {state, idx, val, reward};
}

// Finds the sampled row whose reward matches `reward` (each test gives every
// transition a unique reward specifically so it can be used as a fingerprint to
// find a transition again after sample()'s internal shuffle).
int find_row_by_reward(const torch::Tensor &rewards, float reward) {
    for (int64_t i = 0; i < rewards.size(0); i++) {
        if (rewards[i].item<float>() == reward) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace

TEST_GROUP(ReplayBufferTests){void setup(){} void teardown(){}};

TEST(ReplayBufferTests, SampleExpandsSparsePolicyToCorrectDenseRow) {
    spdlog::info("Testing that sample() expands a sparse policy to the correct dense row...");
    ReplayBuffer buffer(4, kActionSize);
    buffer.add({make_transition(/*state_fill=*/1.0f, {1, 4}, {0.25f, 0.75f}, /*reward=*/42.0f)});

    auto [states, policies, rewards] = buffer.get_sampler().sample(1);

    CHECK_EQUAL(1, states.size(0));
    CHECK_EQUAL(kActionSize, policies.size(1));
    DOUBLES_EQUAL(1.0, states[0][0].item<float>(), 1e-6);
    DOUBLES_EQUAL(42.0, rewards[0].item<float>(), 1e-6);

    for (int64_t a = 0; a < kActionSize; a++) {
        float expected = 0.0f;
        if (a == 1) {
            expected = 0.25f;
        } else if (a == 4) {
            expected = 0.75f;
        }
        DOUBLES_EQUAL(expected, policies[0][a].item<float>(), 1e-6);
    }
}

TEST(ReplayBufferTests, OverwrittenSlotNeverReturnsStaleCachedPolicy) {
    spdlog::info("Testing that overwriting a slot invalidates its cached dense policy...");
    ReplayBuffer buffer(2, kActionSize);

    buffer.add({make_transition(1.0f, {0}, {1.0f}, /*reward=*/100.0f)}); // slot 0
    buffer.add({make_transition(2.0f, {1}, {1.0f}, /*reward=*/200.0f)}); // slot 1

    auto sampler = buffer.get_sampler();
    sampler.sample(2); // forces both slots' dense rows into the cache

    // Circular buffer: capacity=2, two prior adds already wrapped ptr back to 0,
    // so this add lands on (and overwrites) slot 0 - the one whose dense row is
    // already cached above.
    buffer.add({make_transition(3.0f, {2}, {1.0f}, /*reward=*/300.0f)});

    auto [states, policies, rewards] = sampler.sample(2);

    int row_300 = find_row_by_reward(rewards, 300.0f);
    int row_200 = find_row_by_reward(rewards, 200.0f);
    CHECK_TRUE(row_300 != -1);
    CHECK_TRUE(row_200 != -1);
    CHECK_TRUE(find_row_by_reward(rewards, 100.0f) == -1); // slot 0's old data is gone

    // The overwritten slot must reflect transition C (index 2), never stale data
    // from transition A (index 0) - if add() didn't invalidate the cache, this
    // would still show a 1.0 at index 0 instead of index 2.
    DOUBLES_EQUAL(3.0, states[row_300][0].item<float>(), 1e-6);
    DOUBLES_EQUAL(0.0, policies[row_300][0].item<float>(), 1e-6);
    DOUBLES_EQUAL(1.0, policies[row_300][2].item<float>(), 1e-6);

    // The untouched slot's cached row should still be intact and correct.
    DOUBLES_EQUAL(2.0, states[row_200][0].item<float>(), 1e-6);
    DOUBLES_EQUAL(1.0, policies[row_200][1].item<float>(), 1e-6);
}

TEST(ReplayBufferTests, NewSamplerAfterPriorOneDestroyedRecomputesCorrectly) {
    spdlog::info("Testing that a sampler's destructor cleanup doesn't corrupt later samples...");
    ReplayBuffer buffer(2, kActionSize);
    buffer.add({make_transition(1.0f, {3}, {0.5f}, /*reward=*/7.0f)});

    {
        auto sampler = buffer.get_sampler();
        sampler.sample(1); // populate the cache; destructor clears it at the closing brace
    }

    // A fresh sampler, obtained after the previous one was destroyed, must still
    // recompute the correct dense row from scratch.
    auto [states, policies, rewards] = buffer.get_sampler().sample(1);
    DOUBLES_EQUAL(7.0, rewards[0].item<float>(), 1e-6);
    DOUBLES_EQUAL(0.5, policies[0][3].item<float>(), 1e-6);
    for (int64_t a = 0; a < kActionSize; a++) {
        if (a != 3) {
            DOUBLES_EQUAL(0.0, policies[0][a].item<float>(), 1e-6);
        }
    }
}

TEST(ReplayBufferTests, GetSizeGrowsThenSaturatesAtCapacity) {
    spdlog::info("Testing that get_size() saturates at capacity instead of growing past it...");
    ReplayBuffer buffer(2, kActionSize);
    CHECK_EQUAL(0u, buffer.get_size());

    buffer.add({make_transition(1.0f, {0}, {1.0f}, 1.0f)});
    CHECK_EQUAL(1u, buffer.get_size());

    buffer.add({make_transition(2.0f, {0}, {1.0f}, 2.0f)});
    CHECK_EQUAL(2u, buffer.get_size());

    // Buffer is already at capacity - one more add() overwrites a slot rather
    // than growing the reported size further.
    buffer.add({make_transition(3.0f, {0}, {1.0f}, 3.0f)});
    CHECK_EQUAL(2u, buffer.get_size());
}

TEST(ReplayBufferTests, SampleClampsBatchSizeToCurrentSize) {
    spdlog::info("Testing that sample() clamps batch_size to the current size...");
    ReplayBuffer buffer(10, kActionSize);
    buffer.add({make_transition(1.0f, {0}, {1.0f}, 1.0f)});
    buffer.add({make_transition(2.0f, {1}, {1.0f}, 2.0f)});

    auto [states, policies, rewards] = buffer.get_sampler().sample(10);
    CHECK_EQUAL(2, states.size(0));
    CHECK_EQUAL(2, policies.size(0));
    CHECK_EQUAL(2, rewards.size(0));
}

TEST(ReplayBufferTests, TransitionWithNoVisitedActionsProducesAnAllZeroRow) {
    spdlog::info("Testing that an empty sparse policy densifies to an all-zero row...");
    ReplayBuffer buffer(1, kActionSize);
    buffer.add({make_transition(1.0f, {}, {}, 5.0f)});

    auto [states, policies, rewards] = buffer.get_sampler().sample(1);
    for (int64_t a = 0; a < kActionSize; a++) {
        DOUBLES_EQUAL(0.0, policies[0][a].item<float>(), 1e-6);
    }
}

TEST(ReplayBufferTests, StaysCorrectUnderRepeatedEvictionChurn) {
    spdlog::info("Testing that repeated sampling past max_cache_entries stays correct...");
    constexpr size_t kNumSlots = 20;
    // max_cache_entries=3 is far smaller than the 20 distinct slots each
    // sample() call below draws, so every round forces continuous eviction
    // via the same persistent sampler (reused across rounds, unlike other
    // tests here, specifically to exercise that path).
    ReplayBuffer buffer(kNumSlots, kActionSize, /*max_cache_entries=*/3);

    for (size_t i = 0; i < kNumSlots; i++) {
        auto action = static_cast<int64_t>(i % kActionSize);
        buffer.add(
            {make_transition(static_cast<float>(i), {action}, {1.0f}, static_cast<float>(i))});
    }

    auto sampler = buffer.get_sampler();
    for (int round = 0; round < 10; round++) {
        auto [states, policies, rewards] = sampler.sample(kNumSlots);
        for (int64_t row = 0; row < rewards.size(0); row++) {
            int64_t expected_action =
                static_cast<int64_t>(rewards[row].item<float>()) % kActionSize;
            for (int64_t a = 0; a < kActionSize; a++) {
                float expected = (a == expected_action) ? 1.0f : 0.0f;
                DOUBLES_EQUAL(expected, policies[row][a].item<float>(), 1e-6);
            }
        }
    }
}

int main(int ac, char **av) {
    MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
    return CommandLineTestRunner::RunAllTests(ac, av);
}
