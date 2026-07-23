#include "inference/dual_inferer.hpp"
#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>
#include <vector>

struct MockInferer : public Inferer {
    float value_to_return;
    std::vector<int> actions_to_return;
    std::vector<float> logits_to_return;

    MockInferer(float v, std::vector<int> a, std::vector<float> l)
        : Inferer(torch::kCPU), value_to_return(v), actions_to_return(std::move(a)),
          logits_to_return(std::move(l)) {}

    std::vector<inference_result> infer(const std::vector<const GameState *> &states) override {
        std::vector<inference_result> res(states.size());
        for (auto &r : res) {
            r.legal_actions = actions_to_return;
            r.legal_action_logits = logits_to_return;
            r.value = value_to_return;
        }
        return res;
    }
};

TEST_GROUP(DualNetworkInfererTests){};

TEST(DualNetworkInfererTests, MergesPolicyAndValue) {
    auto policy_inferer = std::make_unique<MockInferer>(0.1f, std::vector<int>{1, 2},
                                                        std::vector<float>{0.5f, -0.5f});
    auto value_inferer = std::make_unique<MockInferer>(0.9f, std::vector<int>{3, 4, 5},
                                                       std::vector<float>{1.0f, 2.0f, 3.0f});

    DualNetworkInferer dual(std::move(policy_inferer), std::move(value_inferer));

    std::vector<const GameState *> states(2, nullptr);
    auto results = dual.infer(states);

    CHECK_EQUAL(2, results.size());
    for (const auto &res : results) {
        CHECK_EQUAL(2, res.legal_actions.size());
        CHECK_EQUAL(1, res.legal_actions[0]);
        CHECK_EQUAL(2, res.legal_actions[1]);

        CHECK_EQUAL(2, res.legal_action_logits.size());
        CHECK_EQUAL(0.5f, res.legal_action_logits[0]);
        CHECK_EQUAL(-0.5f, res.legal_action_logits[1]);

        CHECK_EQUAL(0.9f, res.value);
    }
}

int main(int argc, char **argv) {
    return RUN_ALL_TESTS(argc, argv);
}
