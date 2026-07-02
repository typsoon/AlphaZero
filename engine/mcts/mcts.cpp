#include "mcts.hpp"
#include "basic_inferer.hpp"

#include <algorithm>
#include <c10/core/Device.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>
#include <unordered_map>

#include <random>

// TODO: test as many methods as you can
struct MCTS::Node {
    Node **children;
    int action_size;
    Node *parent = nullptr;
    int visits = 0;
    float value = 0.0f;
    float prior = 0.0f;
    bool expanded = false;
    bool is_terminal = false;
    float reward = 0.0f;
    int virtual_loss_count = 0;
    int *valid_actions = nullptr;
    int valid_action_count = 0;
    static constexpr float VL = 1.0f;

    Node(int action_size, std::pmr::memory_resource *pool, float prior_value = 0.0f,
         Node *parent_node = nullptr, bool terminal = false, float rew = 0.0f);

    float Q() const;
    float UCB(float exploration_weight) const;

    void expand(const std::vector<std::pair<int, float>> &policy, std::pmr::memory_resource *pool);

    bool terminal() const;
    bool is_expanded() const;

    // TODO: test this
    std::pair<int, Node *> select_child(float exploration_weight) const;

    // TODO: test this
    static void backpropagate(Node *node, float value, bool vloss = false);
};

void MCTS::Node::backpropagate(Node *node, float value, bool vloss) {
    while (node != nullptr) {
        node->visits++;
        node->value += value;
        if (vloss)
            node->virtual_loss_count--;
        node = node->parent;
        value = -value;
    }
}

MCTS::Node::Node(int action_size, std::pmr::memory_resource * /*pool*/, float prior_value,
                 Node *parent_node, bool terminal, float rew) // NOLINT
    : action_size(action_size), parent(parent_node), prior(prior_value), is_terminal(terminal),
      children(nullptr), reward(rew), valid_actions(nullptr), valid_action_count(0) {
    // We defer the allocation of the `children` array until `expand()` is called.
    // Unexpanded leaf nodes do not need a children array. This saves ~1.1MB per expansion!
}

float MCTS::Node::Q() const {
    // return visits > 0 ? value / visits : 0.0f;
    float adj_value = value + (virtual_loss_count * VL);
    int adj_visits = visits + virtual_loss_count;
    return adj_visits > 0 ? adj_value / adj_visits : 0.0f;
}

float MCTS::Node::UCB(float exploration_weight) const {
    if (parent == nullptr)
        return 0.0f;
    return -Q() + (exploration_weight * prior *
                   std::sqrt(parent->visits + parent->virtual_loss_count + 1) /
                   (1 + visits + virtual_loss_count));
}

void MCTS::Node::expand(const std::vector<std::pair<int, float>> &policy,
                        std::pmr::memory_resource *pool) {
    // Allocate the children array now that the node is actually being expanded.
    children = static_cast<Node **>(pool->allocate(action_size * sizeof(Node *), alignof(Node *)));
    std::memset(static_cast<void *>(children), 0, action_size * sizeof(Node *));

    valid_actions = static_cast<int *>(pool->allocate(policy.size() * sizeof(int), alignof(int)));
    int idx = 0;

    std::pmr::polymorphic_allocator<Node> alloc(pool);
    for (const auto &pair : policy) {
        int i = pair.first;
        float p = pair.second;
        if (p == 0.0f)
            continue;

        Node *child = alloc.allocate(1);
        alloc.construct(child, action_size, pool, p, this, false, 0.0f);
        children[i] = child;
        valid_actions[idx++] = i;
    }
    valid_action_count = idx;
    expanded = true;
}

bool MCTS::Node::terminal() const {
    return is_terminal;
}

bool MCTS::Node::is_expanded() const {
    return expanded;
}

std::pair<int, MCTS::Node *> MCTS::Node::select_child(float exploration_weight) const { // NOLINT
    int best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < valid_action_count; ++k) {
        int i = valid_actions[k];
        float ucb_value = children[i]->UCB(exploration_weight);
        if (ucb_value > best_value) {
            best_value = ucb_value;
            best_index = i;
        }
    }
    return {best_index, best_index != -1 ? children[best_index] : nullptr};
}

MCTS::MCTS(unique_ptr<Inferer> &&network_ptr, float c_init, float c_base, float eps, float alpha,
           size_t arena_size_bytes) // NOLINT
    : network(std::move(network_ptr)), c_init(c_init), c_base(c_base), eps(eps), alpha(alpha),
      device(this->network->device), arena_buffer(arena_size_bytes),
      pool(arena_buffer.data(), arena_buffer.size()) {}

MCTS::MCTS(std::string network_path, torch::Device device, float c_init, float c_base,
           float eps, // NOLINT
           float alpha, size_t arena_size_bytes)
    : network([&device, &network_path]() {
          auto network_inferer_factory = NetworkInfererFactory(network_path, device);
          return network_inferer_factory.get_inferer();
      }()),
      c_init(c_init), c_base(c_base), eps(eps), alpha(alpha), device(this->network->device),
      arena_buffer(arena_size_bytes), pool(arena_buffer.data(), arena_buffer.size()) {}

void MCTS::evaluate_batch(std::vector<std::pair<Node *, std::shared_ptr<Game>>> &leaves,
                          std::pmr::memory_resource *pool) { // NOLINT
    if (leaves.empty())
        return;

    // Multiple simulations in one round can walk down to the same not-yet-expanded
    // node - expansion only happens after the whole round is collected here, so two
    // simulations that both reach it first both see it as unexpanded and both get
    // added as leaves. That's the *only* way a leaf can show up already-expanded in
    // the loop below (a node from an earlier round is never re-added - the tree walk
    // always stops at the first unexpanded node). Since both occurrences are the
    // identical board position, dedupe by Node* so inference only runs once per
    // unique position; each occurrence still backpropagates on its own below, since
    // each carries its own virtual loss from selection.
    std::vector<const GameState *> states;
    std::vector<size_t> result_index(leaves.size());
    std::unordered_map<Node *, size_t> seen;
    seen.reserve(leaves.size());
    for (size_t i = 0; i < leaves.size(); ++i) {
        Node *node = leaves[i].first;
        auto [it, inserted] = seen.try_emplace(node, states.size());
        result_index[i] = it->second;
        if (inserted) {
            states.push_back(leaves[i].second->get_canonical_state().get());
        }
    }

    auto outputs = network->infer(states);

    for (size_t i = 0; i < leaves.size(); ++i) {
        Node *node = leaves[i].first;
        const auto &res = outputs[result_index[i]];

        if (node->is_expanded()) {
            Node::backpropagate(node, res.value, true);
            continue;
        }

        auto policy = get_policy_from_logits(res, false);

        node->expand(policy, pool);
        Node::backpropagate(node, res.value, true);
    }
}

std::pair<std::vector<float>, float> MCTS::search(const Game &game, int num_simulations, // NOLINT
                                                  int batch_size) {
    pool.release();
    Node root_node(game.getActionSize(), &pool, 0.0f, nullptr, game.is_terminal(), game.reward());

    auto inference_res =
        network->infer(std::vector<const GameState *>{game.get_canonical_state().get()});
    float root_value = inference_res.front().value;
    auto p_init = get_policy_from_logits(inference_res.front(), true);
    root_node.expand(p_init, &pool);

    int simulations_done = 0;
    std::vector<std::pair<Node *, std::shared_ptr<Game>>> leaves;
    while (simulations_done < num_simulations) {
        leaves.clear();

        for (int b = 0; b < batch_size && simulations_done < num_simulations;
             ++b, ++simulations_done) {
            Node *node = &root_node;
            auto current_game = game.clone();
            float c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;

            while (node->is_expanded() && !node->terminal()) {
                auto [best_action, best_child] = node->select_child(c_puct);
                node->virtual_loss_count++;
                node = best_child;
                current_game->step(best_action);

                if (node->visits == 0 && current_game->is_terminal()) {
                    node->is_terminal = true;
                    node->reward = current_game->reward();
                }
                c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;
            }

            node->virtual_loss_count++;

            if (!node->terminal()) {
                leaves.emplace_back(node, std::move(current_game));
            } else {
                Node::backpropagate(node, -node->reward, true);
            }
        }

        evaluate_batch(leaves, &pool);
    }

    int A = game.getActionSize();
    std::vector<float> policy(A, 0.0f);
    std::vector<float> pi(A, 0.0f);
    for (int a = 0; a < A; a++) {
        if (root_node.children[a] != nullptr)
            pi[a] = static_cast<float>(root_node.children[a]->visits);
    }
    float sum = std::accumulate(pi.begin(), pi.end(), 0.0f);
    if (sum > 0.0f)
        for (auto &x : pi)
            x /= sum;
    return {pi, root_value};
}

std::vector<std::pair<int, float>> MCTS::get_policy_from_logits(const inference_result &res,
                                                                bool dirichletNoise) const {
    // We omit libtorch (ATen) operations here (like torch::tensor, torch::softmax)
    // because the ATen dispatcher overhead and allocations are extremely slow for
    // small vectors on the CPU (e.g., 7 elements for Connect4). A pure C++
    // implementation is ~6x faster and avoids creating intermediate tensors.
    // It operates on the same 32-bit floats natively, so no precision is lost.

    // res.legal_actions[j]/res.legal_action_logits[j] are computed by Inferer itself
    // (via GameState::get_legal_actions()), so no separate legal-actions lookup is
    // needed here.
    const auto &legal_actions = res.legal_actions;
    std::vector<std::pair<int, float>> policy_vec;
    policy_vec.reserve(legal_actions.size());

    float max_logit = -std::numeric_limits<float>::infinity();
    for (float logit : res.legal_action_logits) {
        max_logit = std::max(logit, max_logit);
    }

    float sum_exp = 0.0f;
    for (size_t j = 0; j < legal_actions.size(); ++j) {
        float p = std::exp(res.legal_action_logits[j] - max_logit);
        policy_vec.emplace_back(legal_actions[j], p);
        sum_exp += p;
    }

    if (sum_exp > 0.0f) {
        for (auto &pair : policy_vec) {
            pair.second /= sum_exp;
        }
    }

    if (dirichletNoise) {
        std::vector<float> alpha_vec(legal_actions.size(), alpha);
        auto noise_vec = sample_dirichlet(alpha_vec);

        for (size_t i = 0; i < policy_vec.size(); ++i) {
            policy_vec[i].second = ((1.0f - eps) * policy_vec[i].second) + (eps * noise_vec[i]);
        }
    }

    return policy_vec;
}

std::vector<float> MCTS::sample_dirichlet(const std::vector<float> &alpha) { // NOLINT
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::vector<float> x(alpha.size());
    float sum = 0.0f;
    for (size_t i = 0; i < alpha.size(); ++i) {
        std::gamma_distribution<float> dist(alpha[i], 1.0f);
        x[i] = dist(gen);
        sum += x[i];
    }
    if (sum > 0.0f) {
        for (auto &v : x)
            v /= sum;
    }
    return x;
}
