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

struct MCTS::Node {
    // Fields ordered largest-alignment-first (8-byte pointers, then 4-byte ints/
    // floats, then the two bools) to minimize padding - this struct is allocated
    // out of the arena up to once per simulation (see default_arena_size_in_bytes
    // above), so its size directly multiplies into the arena's per-search memory
    // budget. The original declaration order interleaved pointers/ints/bools and
    // cost 64 bytes/node to this order's 56.
    Node **children;
    Node *parent = nullptr;
    // int16_t, not int: indices into an action space that tops out at 20480
    // (chess's action_dim) - well within int16_t's +-32767 range - and this array
    // is sized to the branching factor (policy.size() in expand(), typically
    // ~20-40 legal moves for chess), so halving its element size directly halves
    // one of the few genuinely per-node-count-scaled arena allocations.
    int16_t *valid_actions = nullptr;
    int action_size;
    // Deliberately left as int (not narrowed like the fields below): visits is
    // driven by num_simulations, a caller-supplied, effectively unbounded search()
    // parameter (see the arena-sizing comment above default_arena_size_in_bytes,
    // which already documents simulation counts well past the 800 default as a
    // supported case) - int16_t's 32767 cap would silently wrap and corrupt search
    // results for any caller who raises it past that, with no compiler warning.
    int visits = 0;
    float value = 0.0f;
    float prior = 0.0f;
    float reward = 0.0f;
    // Bounded by mcts_batch_size (concurrent in-flight traversals through this
    // node within one batch round, before backpropagate() resolves them) - orders
    // of magnitude under int16_t's range even for large batch sizes.
    int16_t virtual_loss_count = 0;
    // Bounded by the branching factor, same as valid_actions above.
    int16_t valid_action_count = 0;
    bool expanded = false;
    bool is_terminal = false;
    static constexpr float VL = 1.0f;

    Node(int action_size, std::pmr::memory_resource *pool, float prior_value = 0.0f,
         Node *parent_node = nullptr, bool terminal = false, float rew = 0.0f);

    float Q() const;
    float UCB(float exploration_weight, float fpu_reduction) const;

    void expand(const std::vector<std::pair<int, float>> &policy, std::pmr::memory_resource *pool,
                bool zero_children = false);

    bool terminal() const;
    bool is_expanded() const;

    std::pair<int, Node *> select_child(float exploration_weight, float fpu_reduction) const;

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
    : children(nullptr), parent(parent_node), valid_actions(nullptr), action_size(action_size),
      prior(prior_value), reward(rew), valid_action_count(0), is_terminal(terminal) {
    // We defer the allocation of the `children` array until `expand()` is called.
    // Unexpanded leaf nodes do not need a children array. This saves ~1.1MB per expansion!
}

float MCTS::Node::Q() const {
    // return visits > 0 ? value / visits : 0.0f;
    float adj_value = value + (virtual_loss_count * VL);
    int adj_visits = visits + virtual_loss_count;
    return adj_visits > 0 ? adj_value / adj_visits : 0.0f;
}

float MCTS::Node::UCB(float exploration_weight, float fpu_reduction) const {
    if (parent == nullptr)
        return 0.0f;
    // Exploitation term (parent-to-move perspective): a visited child uses its
    // own backed-up value, -Q(); an unvisited child uses first-play urgency -
    // the parent's own running value minus fpu_reduction - instead of the
    // implicit "assume draw" 0.0. parent->Q() is the parent's mean from the
    // parent's perspective, the same convention this exploitation term is in.
    float exploit = (visits + virtual_loss_count) > 0 ? -Q() : (parent->Q() - fpu_reduction);
    return exploit + (exploration_weight * prior *
                      std::sqrt(parent->visits + parent->virtual_loss_count + 1) /
                      (1 + visits + virtual_loss_count));
}

void MCTS::Node::expand(const std::vector<std::pair<int, float>> &policy,
                        std::pmr::memory_resource *pool, bool zero_children) {
    // Allocate the children array now that the node is actually being expanded.
    children = static_cast<Node **>(pool->allocate(action_size * sizeof(Node *), alignof(Node *)));
    // Non-root nodes are only ever indexed via valid_actions (select_child()), so unwritten
    // slots are never read; only the root's array is scanned in full (search()'s pi loop).
    if (zero_children)
        std::memset(static_cast<void *>(children), 0, action_size * sizeof(Node *));

    valid_actions =
        static_cast<int16_t *>(pool->allocate(policy.size() * sizeof(int16_t), alignof(int16_t)));
    int idx = 0;

    std::pmr::polymorphic_allocator<Node> alloc(pool);
    for (const auto &pair : policy) {
        int i = pair.first;
        float p = pair.second;
        // Skipping a 0-probability legal action here (rather than allocating a Node
        // for it) is the reason valid_actions/valid_action_count exist at all -
        // every reader of this tree (select_child(), search()'s root-pi loop,
        // search_gumbel()) must go through valid_actions, not iterate legal_actions
        // and assume children[a] is populated. See search_gumbel()'s use of
        // root_node.valid_actions (not the raw legal_actions from inference) for why
        // that distinction matters there specifically.
        if (p == 0.0f)
            continue;

        Node *child = alloc.allocate(1);
        alloc.construct(child, action_size, pool, p, this, false, 0.0f);
        children[i] = child;
        valid_actions[idx++] = static_cast<int16_t>(i);
    }
    valid_action_count = static_cast<int16_t>(idx);
    expanded = true;
}

bool MCTS::Node::terminal() const {
    return is_terminal;
}

bool MCTS::Node::is_expanded() const {
    return expanded;
}

std::pair<int, MCTS::Node *> MCTS::Node::select_child(float exploration_weight, // NOLINT
                                                      float fpu_reduction) const {
    int best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < valid_action_count; ++k) {
        int i = valid_actions[k];
        float ucb_value = children[i]->UCB(exploration_weight, fpu_reduction);
        if (ucb_value > best_value) {
            best_value = ucb_value;
            best_index = i;
        }
    }
    return {best_index, best_index != -1 ? children[best_index] : nullptr};
}

MCTS::MCTS(unique_ptr<Inferer> &&network_ptr, float c_init, float c_base, float eps, float alpha,
           size_t arena_size_bytes, float fpu_reduction) // NOLINT
    : network(std::move(network_ptr)), c_init(c_init), c_base(c_base), eps(eps), alpha(alpha),
      fpu_reduction(fpu_reduction), device(this->network->device), arena_buffer(arena_size_bytes),
      pool(arena_buffer.data(), arena_buffer.size()) {}

MCTS::MCTS(std::string network_path, torch::Device device, float c_init, float c_base,
           float eps, // NOLINT
           float alpha, size_t arena_size_bytes, float fpu_reduction,
           std::shared_ptr<StateEncoder> encoder)
    : network([&device, &network_path, &encoder]() {
          auto network_inferer_factory =
              NetworkInfererFactory(network_path, device, /*wait_for_count=*/1, /*timeout_ms=*/10,
                                    /*transposition_cache_entries=*/1000000, std::move(encoder));
          return network_inferer_factory.get_inferer();
      }()),
      c_init(c_init), c_base(c_base), eps(eps), alpha(alpha), fpu_reduction(fpu_reduction),
      device(this->network->device), arena_buffer(arena_size_bytes),
      pool(arena_buffer.data(), arena_buffer.size()) {}

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
    root_node.expand(p_init, &pool, /*zero_children=*/true);

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
                auto [best_action, best_child] = node->select_child(c_puct, fpu_reduction);
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
                // node->reward() is already expressed in the same "value for the
                // player to move at this node" convention that backpropagate expects
                // (see evaluate_batch(), which passes res.value straight through) -
                // it must not be re-negated here.
                Node::backpropagate(node, node->reward, true);
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

std::vector<float> MCTS::sample_gumbel(size_t n) { // NOLINT
    static thread_local std::mt19937 gen{std::random_device{}()};
    // Open interval (0,1): std::log(-std::log(u)) is undefined at either endpoint
    // (u=0 -> log(0); u=1 -> log(-log(1))=log(0)).
    std::uniform_real_distribution<float> dist(std::numeric_limits<float>::epsilon(),
                                               1.0f - std::numeric_limits<float>::epsilon());
    std::vector<float> g(n);
    for (size_t i = 0; i < n; ++i) {
        g[i] = -std::log(-std::log(dist(gen)));
    }
    return g;
}

MCTS::gumbel_result MCTS::search_gumbel(const Game &game, int num_simulations, // NOLINT
                                        int batch_size, int max_num_considered_actions,
                                        float c_visit, float c_scale) {
    pool.release();
    Node root_node(game.getActionSize(), &pool, 0.0f, nullptr, game.is_terminal(), game.reward());

    auto inference_res =
        network->infer(std::vector<const GameState *>{game.get_canonical_state().get()});
    const auto &root_res = inference_res.front();
    float root_value = root_res.value;

    // Root expansion still needs a prior distribution for PUCT to use below the
    // root - Gumbel-Top-k below is this method's own root-level exploration
    // mechanism, so (unlike search()) no Dirichlet noise is mixed in here.
    auto p_init = get_policy_from_logits(root_res, /*dirichletNoise=*/false);
    root_node.expand(p_init, &pool, /*zero_children=*/true);

    int A = game.getActionSize();
    const auto &legal_actions = root_res.legal_actions;
    const auto &legal_logits = root_res.legal_action_logits;
    int num_legal = static_cast<int>(legal_actions.size());

    std::vector<float> pi(A, 0.0f);

    // Nothing to search with at most one legal move - matches the paper, which
    // skips planning entirely in this case.
    if (num_legal <= 1) {
        if (num_legal == 1)
            pi[legal_actions[0]] = 1.0f;
        return {pi, root_value, num_legal == 1 ? legal_actions[0] : -1};
    }

    std::vector<float> logit_of(A, 0.0f);
    std::vector<float> gumbel_of(A, 0.0f);
    auto gumbel_draws = sample_gumbel(static_cast<size_t>(num_legal));
    for (int j = 0; j < num_legal; ++j) {
        logit_of[legal_actions[j]] = legal_logits[j];
        gumbel_of[legal_actions[j]] = gumbel_draws[j];
    }

    // Built from root_node.valid_actions, not the raw legal_actions from
    // inference: expand() (see its own comment) skips allocating a child for any
    // legal action whose prior underflowed to exactly 0.0f, so valid_actions is
    // the authoritative "legal actions that actually have a root_node.children[a]
    // to index" list - the same one select_child()/search() already trust.
    // considered only ever holds entries from here, so every root_node.children[a]
    // access below (completed_q(), the sequential-halving descent) is safe by
    // construction, without needing its own null check.
    int valid_action_count = root_node.valid_action_count;
    int m = std::min(max_num_considered_actions, valid_action_count);
    std::vector<int> considered(root_node.valid_actions,
                                root_node.valid_actions + valid_action_count);
    // Gumbel-Top-k (Kool, van Hoof & Welling, 2019): taking the top m actions by
    // g_a + logit_a is equivalent in distribution to sampling m actions without
    // replacement from softmax(logits) - but reuses the same g_a draws for every
    // later sequential-halving cut below.
    std::partial_sort(
        considered.begin(), considered.begin() + m, considered.end(),
        [&](int a, int b) { return (gumbel_of[a] + logit_of[a]) > (gumbel_of[b] + logit_of[b]); });
    considered.resize(m);

    // sigma() rescales a completed Q-value onto the same scale as raw policy
    // logits, weighted by how much visit evidence the root already has - the
    // paper's completedQ transform (Sec. 3), c_visit=50/c_scale=1.0 board-game
    // defaults. Following the authors' reference implementation (mctx's
    // qtransform_completed_by_mix_value with rescale_values=True), q is first
    // min-max normalized to [0, 1] over the completed Qs of all root children
    // (seeded with v_mix itself, the completed Q every unvisited child shares):
    // without the rescale, sigma's weight relative to the raw logits collapses
    // whenever the children's Q spread is small (quiet positions), leaving the
    // improved policy target barely sharpened over the prior. The transform's
    // inputs (max visits, min, 1/range) are computed once per use site via
    // root_q_bounds() rather than inside sigma(): sigma() is evaluated inside
    // sort comparators below, and none of them change while a sort runs.
    struct QBounds {
        float max_child_visits;
        float min_q;
        float inv_range;
    };
    auto root_q_bounds = [&](float v_mix) {
        QBounds b{0.0f, v_mix, 0.0f};
        float max_q = v_mix;
        for (int k = 0; k < root_node.valid_action_count; ++k) {
            int a = root_node.valid_actions[k];
            Node *child = root_node.children[a];
            b.max_child_visits = std::max(b.max_child_visits, static_cast<float>(child->visits));
            float q = child->visits > 0 ? -child->Q() : v_mix;
            b.min_q = std::min(b.min_q, q);
            max_q = std::max(max_q, q);
        }
        float range = max_q - b.min_q;
        constexpr float q_epsilon = 1e-8f;
        if (range > q_epsilon)
            b.inv_range = 1.0f / range;
        return b;
    };
    auto sigma = [&](float q, const QBounds &bounds) {
        float normalized = (q - bounds.min_q) * bounds.inv_range;
        return (c_visit + bounds.max_child_visits) * c_scale * normalized;
    };

    // v_mix: a single value estimate, shared by every not-yet-visited action,
    // blending the root network's own value with the (prior-weighted) backed-up Q
    // of whichever children have been visited so far (paper Sec. 3, "value
    // estimate"). With zero visited children this reduces to the raw root value.
    auto compute_v_mix = [&]() {
        float sum_n = 0.0f;
        float sum_prior = 0.0f;
        float weighted_q = 0.0f;
        for (int k = 0; k < root_node.valid_action_count; ++k) {
            int a = root_node.valid_actions[k];
            Node *child = root_node.children[a];
            if (child->visits > 0) {
                float q_root_perspective = -child->Q();
                sum_n += static_cast<float>(child->visits);
                sum_prior += child->prior;
                weighted_q += child->prior * q_root_perspective;
            }
        }
        if (sum_prior > 0.0f)
            weighted_q /= sum_prior;
        return (root_value + sum_n * weighted_q) / (1.0f + sum_n);
    };

    // completedQ(a): the actual backed-up Q for a visited action, or the shared
    // v_mix estimate for one that sequential halving hasn't simulated (yet).
    auto completed_q = [&](int a, float v_mix) {
        Node *child = root_node.children[a];
        return child->visits > 0 ? -child->Q() : v_mix;
    };

    int simulations_done = 0;
    int phases = static_cast<int>(std::ceil(std::log2(static_cast<double>(m))));
    std::vector<std::pair<Node *, std::shared_ptr<Game>>> leaves;

    for (int phase = 0;
         phase < phases && considered.size() > 1 && simulations_done < num_simulations; ++phase) {
        int n_phase = static_cast<int>(considered.size());
        int extra_visits = std::max(1, num_simulations / (phases * n_phase));

        leaves.clear();
        for (int a : considered) {
            for (int v = 0; v < extra_visits && simulations_done < num_simulations;
                 ++v, ++simulations_done) {
                Node *node = &root_node;
                auto current_game = game.clone();

                // Sequential halving's job is only the root's action choice - force
                // it to `a`, then fall straight into the same PUCT descent search()
                // uses for every level below that.
                node->virtual_loss_count++;
                node = root_node.children[a];
                current_game->step(a);
                if (node->visits == 0 && current_game->is_terminal()) {
                    node->is_terminal = true;
                    node->reward = current_game->reward();
                }

                float c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;
                while (node->is_expanded() && !node->terminal()) {
                    auto [best_action, best_child] = node->select_child(c_puct, fpu_reduction);
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
                    if (static_cast<int>(leaves.size()) >= batch_size) {
                        evaluate_batch(leaves, &pool);
                        leaves.clear();
                    }
                } else {
                    Node::backpropagate(node, node->reward, true);
                }
            }
        }
        if (!leaves.empty()) {
            evaluate_batch(leaves, &pool);
            leaves.clear();
        }

        // Halve: keep the top half of `considered`, ranked by the same g_a+logit_a
        // score used for the initial Top-k cut, now augmented with each action's
        // completedQ evidence from the simulations just run.
        float v_mix = compute_v_mix();
        auto bounds = root_q_bounds(v_mix);
        std::sort(considered.begin(), considered.end(), [&](int a, int b) {
            float score_a = gumbel_of[a] + logit_of[a] + sigma(completed_q(a, v_mix), bounds);
            float score_b = gumbel_of[b] + logit_of[b] + sigma(completed_q(b, v_mix), bounds);
            return score_a > score_b;
        });
        size_t keep = std::max<size_t>(1, (considered.size() + 1) / 2);
        considered.resize(keep);
    }

    // Improved policy target (paper Sec. 3): softmax over all legal actions of
    // logit + sigma(completedQ), *without* the Gumbel noise term - the noise's job
    // was only to pick which actions got simulated, not to bias the final target.
    // Actions sequential halving never got to keep completedQ = v_mix, same as any
    // other unvisited action. Iterates valid_actions, not legal_actions, for the
    // same children[a]-safety reason as `considered` above - any legal action
    // skipped by expand() would score effectively -inf here anyway (that's why its
    // prior underflowed to 0 in the first place), so excluding it changes nothing
    // about the resulting distribution over the actions that *do* get mass.
    float v_mix_final = compute_v_mix();
    auto bounds_final = root_q_bounds(v_mix_final);
    std::vector<float> scores(valid_action_count);
    float max_score = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < valid_action_count; ++k) {
        int a = root_node.valid_actions[k];
        scores[k] = logit_of[a] + sigma(completed_q(a, v_mix_final), bounds_final);
        max_score = std::max(max_score, scores[k]);
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < valid_action_count; ++k) {
        float p = std::exp(scores[k] - max_score);
        pi[root_node.valid_actions[k]] = p;
        sum_exp += p;
    }
    if (sum_exp > 0.0f)
        for (int k = 0; k < valid_action_count; ++k)
            pi[root_node.valid_actions[k]] /= sum_exp;

    // The action to play is the sequential-halving winner: considered is sorted
    // by g + logit + sigma(completedQ) at the end of every phase (and by
    // g + logit when the loop never ran, e.g. m == 1 or a zero simulation
    // budget - which is exactly the paper's selection rule for n = 0 too), so
    // its front is argmax over the survivors. This is deliberately *not*
    // argmax(pi): pi drops the Gumbel term (see the improved-policy comment
    // above), and only the g-inclusive argmax carries the paper's
    // policy-improvement guarantee for the played move.
    return {pi, root_value, considered.front()};
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
