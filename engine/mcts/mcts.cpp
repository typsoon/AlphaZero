#include "mcts.hpp"
#include "basic_inferer.hpp"

#include <c10/core/Device.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>

#include <random>

struct MCTS::Node {
    using NodePtr = std::unique_ptr<Node>;
    std::shared_ptr<Game> game_state;
    vector<NodePtr> children;
    Node *parent = nullptr;
    int visits = 0;
    float value = 0.0f;
    float prior = 0.0f;
    bool expanded = false;
    int virtual_loss_count = 0;
    static constexpr float VL = 1.0f;

    Node(std::shared_ptr<Game> state, float prior_value = 0.0f, Node *parent_node = nullptr);

    float Q() const;
    float UCB(float exploration_weight) const;

    void expand(const std::vector<float> &policy);

    bool is_terminal() const;
    bool is_expanded() const;

    std::pair<int, Node *> select_child(float exploration_weight) const;

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

MCTS::Node::Node(std::shared_ptr<Game> state, float prior_value, Node *parent_node) // NOLINT
    : game_state(std::move(state)), prior(prior_value), parent(parent_node) {
    children.resize(game_state->getActionSize());
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

void MCTS::Node::expand(const std::vector<float> &policy) {
    for (int i = 0; i < policy.size(); i++) {
        if (policy[i] == 0.0f)
            continue;
        auto child_state = game_state->clone();
        child_state->step(i);

        children[i] = make_unique<Node>(std::move(child_state), policy[i], this);
    }
    expanded = true;
}

bool MCTS::Node::is_terminal() const {
    return game_state->is_terminal();
}

bool MCTS::Node::is_expanded() const {
    return expanded;
}

std::pair<int, MCTS::Node *> MCTS::Node::select_child(float exploration_weight) const { // NOLINT
    int best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < children.size(); i++) {
        if (!children[i])
            continue;
        float ucb_value = 0.0f;
        ucb_value = children[i]->UCB(exploration_weight);
        if (ucb_value > best_value) {
            best_value = ucb_value;
            best_index = i;
        }
    }
    return {best_index, children[best_index].get()};
}

MCTS::MCTS(unique_ptr<Inferer> &&network_ptr, float c_init, float c_base, float eps,
           float alpha) // NOLINT
    : network(std::move(network_ptr)), c_init(c_init), c_base(c_base), eps(eps), alpha(alpha),
      device(this->network->device) {}

MCTS::MCTS(std::string network_path, torch::Device device, float c_init, float c_base,
           float eps, // NOLINT
           float alpha)
    : network([&device, &network_path]() {
          auto network_inferer_factory = NetworkInfererFactory(network_path, device);
          return network_inferer_factory.get_inferer();
      }()),
      c_init(c_init), c_base(c_base), eps(eps), alpha(alpha), device(this->network->device) {}

void MCTS::evaluate_batch(std::vector<Node *> &leaves) { // NOLINT
    if (leaves.empty())
        return;

    std::vector<const GameState *> states;
    states.reserve(leaves.size());
    for (auto *leaf : leaves) {
        states.push_back(leaf->game_state->get_canonical_state().get());
    }

    auto outputs = network->infer(states);

    for (size_t i = 0; i < leaves.size(); ++i) {
        Node *node = leaves[i];
        const auto &[policy_tensor, value] = outputs[i];

        if (node->is_expanded()) {
            Node::backpropagate(node, value, true);
            continue;
        }

        auto policy =
            get_policy_from_logits(policy_tensor, node->game_state->get_legal_actions(), false);

        node->expand(policy);
        Node::backpropagate(node, value, true);
    }
}

std::pair<std::vector<float>, float> MCTS::search(const Game &game, int num_simulations, // NOLINT
                                                  int batch_size) {
    auto root_game = game.clone();
    Node root_node(std::move(root_game));

    auto inference_res = network->infer(
        std::vector<const GameState *>{root_node.game_state->get_canonical_state().get()});
    float root_value = inference_res.front().second;
    auto p_init = get_policy_from_logits(inference_res.front().first,
                                         root_node.game_state->get_legal_actions(), true);
    root_node.expand(p_init);

    int simulations_done = 0;
    std::vector<Node *> leaves;
    while (simulations_done < num_simulations) {
        leaves.clear();

        for (int b = 0; b < batch_size && simulations_done < num_simulations;
             ++b, ++simulations_done) {
            Node *node = &root_node;
            float c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;

            while (node->is_expanded() && !node->is_terminal()) {
                auto [best_action, best_child] = node->select_child(c_puct);
                node->virtual_loss_count++;
                node = best_child;
                c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;
            }

            node->virtual_loss_count++;

            if (!node->is_terminal()) {
                leaves.push_back(node);
            } else {
                Node::backpropagate(node, -node->game_state->reward(), true);
            }
        }

        evaluate_batch(leaves);
    }

    int A = game.getActionSize();
    std::vector<float> policy(A, 0.0f);
    std::vector<float> pi(A, 0.0f);
    for (int a = 0; a < A; a++) {
        if (root_node.children[a])
            pi[a] = static_cast<float>(root_node.children[a]->visits);
    }
    float sum = std::accumulate(pi.begin(), pi.end(), 0.0f);
    if (sum > 0.0f)
        for (auto &x : pi)
            x /= sum;
    return {pi, root_value};
}

std::vector<float> MCTS::get_policy_from_logits(torch::Tensor policy_logits,
                                                const std::vector<int> &legal_actions,
                                                bool dirichletNoise) const {
    // We omit libtorch (ATen) operations here (like torch::tensor, torch::softmax)
    // because the ATen dispatcher overhead and allocations are extremely slow for
    // small vectors on the CPU (e.g., 7 elements for Connect4). A pure C++
    // implementation is ~6x faster and avoids creating intermediate tensors.
    // It operates on the same 32-bit floats natively, so no precision is lost.

    policy_logits = policy_logits.squeeze(0).cpu().contiguous();
    int A = policy_logits.size(0);
    const float *logits_data = policy_logits.data_ptr<float>();

    std::vector<float> policy_vec(A, 0.0f);

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int a : legal_actions) {
        if (logits_data[a] > max_logit) {
            max_logit = logits_data[a];
        }
    }

    float sum_exp = 0.0f;
    for (int a : legal_actions) {
        policy_vec[a] = std::exp(logits_data[a] - max_logit);
        sum_exp += policy_vec[a];
    }

    if (sum_exp > 0.0f) {
        for (int a : legal_actions) {
            policy_vec[a] /= sum_exp;
        }
    }

    if (dirichletNoise) {
        std::vector<float> alpha_vec(legal_actions.size(), alpha);
        auto noise_vec = sample_dirichlet(alpha_vec);

        for (size_t i = 0; i < legal_actions.size(); ++i) {
            int a = legal_actions[i];
            policy_vec[a] = ((1.0f - eps) * policy_vec[a]) + (eps * noise_vec[i]);
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
