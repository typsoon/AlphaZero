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
    unique_ptr<Game> game_state;
    vector<NodePtr> children;
    Node *parent = nullptr;
    int visits = 0;
    float value = 0.0f;
    float prior = 0.0f;
    bool expanded = false;

    Node(std::unique_ptr<Game> state, float prior_value = 0.0f,
         Node *parent_node = nullptr);

    float Q() const;
    float UCB(float exploration_weight) const;

    void expand(const std::vector<float> &policy);

    bool is_terminal() const;
    bool is_expanded() const;

    std::pair<int, Node *> select_child(float exploration_weight) const;

    static void backpropagate(Node *node, float value);
};

void MCTS::Node::backpropagate(Node *node, float value) {
    while (node) {
        node->visits++;
        node->value += value;
        node = node->parent;
        value = -value;
    }
}

MCTS::Node::Node(std::unique_ptr<Game> state, float prior_value,
                 Node *parent_node)
    : game_state(std::move(state)), prior(prior_value), parent(parent_node) {
    children.resize(game_state->getActionSize());
}

float MCTS::Node::Q() const {
    return visits > 0 ? value / visits : 0.0f;
}

float MCTS::Node::UCB(float exploration_weight) const {
    if (!parent)
        return 0;
    return Q() + exploration_weight * prior * std::sqrt(parent->visits) /
                     (1 + visits);
}

void MCTS::Node::expand(const std::vector<float> &policy) {
    for (int i = 0; i < policy.size(); i++) {
        if (policy[i] == 0.0f)
            continue;
        auto child_state = game_state->clone();
        child_state->step(i);

        children[i] =
            make_unique<Node>(std::move(child_state), policy[i], this);
    }
    expanded = true;
}

bool MCTS::Node::is_terminal() const {
    return game_state->is_terminal();
}

bool MCTS::Node::is_expanded() const {
    return expanded;
}

std::pair<int, MCTS::Node *>
MCTS::Node::select_child(float exploration_weight) const {
    int best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < children.size(); i++) {
        if (!children[i])
            continue;
        float ucb_value = children[i]->UCB(exploration_weight);
        if (ucb_value > best_value) {
            best_value = ucb_value;
            best_index = i;
        }
    }
    return {best_index, children[best_index].get()};
}

MCTS::MCTS(unique_ptr<Inferer> &&network, float c_init, float c_base, float eps,
           float alpha)
    : network(std::move(network)), c_init(c_init), c_base(c_base), eps(eps),
      alpha(alpha), device(this->network->device) {}

MCTS::MCTS(std::string network_path, torch::Device device, float c_init,
           float c_base, float eps, float alpha)
    : network([&device, &network_path]() {
          auto network_inferer_factory =
              NetworkInfererFactory(network_path, device);
          return network_inferer_factory.get_inferer();
      }()),
      c_init(c_init), c_base(c_base), eps(eps), alpha(alpha),
      device(this->network->device) {}

void MCTS::evaluate_batch(std::vector<Node *> &leaves) {
    if (leaves.empty())
        return;

    torch::NoGradGuard no_grad;
    std::vector<GameState> inputs;
    inputs.reserve(leaves.size());

    for (auto &leaf : leaves) {
        inputs.emplace_back(leaf->game_state->get_canonical_state());
    }

    auto outputs = network->infer(inputs);

    for (size_t i = 0; i < leaves.size(); ++i) {
        Node *node = leaves[i];
        if (node->is_expanded()) {
            continue;
        }

        const auto &[policy_tensor, value] = outputs[i];
        auto policy = get_policy_from_logits(
            policy_tensor, node->game_state->get_legal_actions(), true);

        node->expand(policy);
        Node::backpropagate(node, value);
    }
}

std::vector<float> MCTS::search(const Game &game, int num_simulations,
                                int batch_size) {
    auto root = game.clone();
    auto inference_res = network->infer({root->get_canonical_state()});
    auto p_init = get_policy_from_logits(inference_res.front().first,
                                         root->get_legal_actions(), true);
    Node root_node = Node(std::move(root));
    root_node.expand(p_init);

    int simulations_done = 0;
    std::vector<Node *> leaves;
    while (simulations_done < num_simulations) {
        leaves.clear();

        for (int b = 0; b < batch_size && simulations_done < num_simulations;
             ++b, ++simulations_done) {
            Node *node = &root_node;
            float c_puct =
                std::log((1 + node->visits + c_base) / c_base) + c_init;

            while (node->is_expanded() && !node->is_terminal()) {
                auto [best_action, best_child] = node->select_child(c_puct);
                node = best_child;
                c_puct =
                    std::log((1 + node->visits + c_base) / c_base) + c_init;
            }

            if (!node->is_terminal()) {
                leaves.push_back(node);
            } else {
                Node::backpropagate(node, node->game_state->reward());
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
    return pi;
}

std::vector<float>
MCTS::get_policy_from_logits(torch::Tensor policy_logits,
                             const std::vector<int> &legal_actions,
                             bool dirichletNoise) {
    policy_logits = policy_logits.squeeze(0); // [A]
    int A = policy_logits.size(0);

    torch::Tensor policy = torch::softmax(policy_logits, 0); // [A]

    if (dirichletNoise) {
        std::vector<float> alpha_vec(legal_actions.size(),
                                     alpha); // alpha to hiperparametr klasy
        auto noise_vec = sample_dirichlet(alpha_vec);

        std::vector<float> full_noise(A, 0.0f);
        for (size_t i = 0; i < legal_actions.size(); ++i)
            full_noise[legal_actions[i]] = noise_vec[i];

        auto noise_t = torch::tensor(full_noise, policy.options());
        policy = (1.0f - eps) * policy + eps * noise_t;
    }

    std::vector<float> mask_vec(A, 0.0f);
    for (int a : legal_actions)
        mask_vec[a] = 1.0f;

    torch::Tensor mask = torch::tensor(mask_vec, policy.options());
    policy = policy * mask;

    float sum = policy.sum().item<float>();
    std::vector<float> policy_vec(A, 0.0f);

    if (sum > 0.0f) {
        policy = policy / sum;
        policy = policy.cpu();
        std::memcpy(policy_vec.data(), policy.data_ptr<float>(),
                    A * sizeof(float));
    } else {
        float inv = 1.0f / static_cast<float>(legal_actions.size());
        for (int a : legal_actions)
            policy_vec[a] = inv;
    }

    return policy_vec;
}
std::vector<float> MCTS::sample_dirichlet(const std::vector<float> &alpha) {
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
