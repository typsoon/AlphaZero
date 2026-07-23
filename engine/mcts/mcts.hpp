#ifndef MCTS_HPP
#define MCTS_HPP

#include "game.hpp"
#include "inferer.hpp"
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <torch/torch.h>
#include <vector>

using std::make_unique;
using std::unique_ptr;
using std::vector;

struct StateEncoder;

// Sized for chess, the larger of the two games: Node::expand() allocates a
// children array sized to the full action space (20480 * 8 bytes = ~160KB) per
// expansion, so at the default 800 simulations/search a single search() call
// needs up to ~130MB (800 * (~160KB children array + ~2KB for the expanded
// child Node structs themselves and their valid_actions array), worst case one
// expansion per simulation). 160MB keeps ~23% headroom above that worst case
// while still mattering a lot at scale: self_play() (see training/self_play.cpp)
// keeps one MCTS - and one arena - alive per OpenMP thread for the whole run,
// so at thread_count=64 this constant alone is responsible for
// thread_count * default_arena_size_in_bytes of self-play's peak memory (was
// 16GiB at the old 256MB default; a documented contributor to two OOM kills of
// the actual training run). If you raise mcts_num_simulations well past 800,
// re-derive this: undersizing it means most expansions overflow into the
// upstream allocator (malloc) instead, which still works but is slower. The
// The arena is fully reset (pool.release()) at the start of every search() call.
inline size_t calculate_arena_size(size_t action_size, int mcts_num_simulations) {
    // 8 bytes per child pointer (children array) + ~2KB for Node/valid_actions struct.
    // 25% safety margin on top.
    return static_cast<size_t>(mcts_num_simulations * (action_size * 8 + 2048) * 1.25);
}
constexpr size_t default_arena_size_in_bytes = static_cast<const size_t>(160 * 1024 * 1024);

class MCTS {
    using InfererPtr = unique_ptr<Inferer>;
    InfererPtr network;
    float c_init;
    float c_base;
    float eps;
    float alpha;
    // First-play-urgency reduction: an unvisited child is scored, in the PUCT
    // exploitation term, at its parent's running value minus this amount
    // instead of the assume-draw 0.0. 0.0 reproduces the original behavior.
    float fpu_reduction;
    torch::Device device;

    std::vector<std::byte> arena_buffer;
    std::pmr::monotonic_buffer_resource pool;

  public:
    MCTS(unique_ptr<Inferer> &&network, float c_init = 1.25f, float c_base = 19652.0f,
         float eps = 0.25f, float alpha = 0.3f,
         size_t arena_size_bytes = default_arena_size_in_bytes, float fpu_reduction = 0.0f);

    // encoder selects the NN input encoding fed to the network at network_path;
    // null (default) derives the game's default via default_encoder_for()
    // (ChessEncoderV1 for chess). Pass a ChessEncoderV2History to serve a
    // history-encoder net (e.g. puzzle-testing the chess-v2 history bootstrap).
    MCTS(std::string network_path, torch::Device device, float c_init = 1.25f,
         float c_base = 19652.0f, float eps = 0.25f, float alpha = 0.3f,
         size_t arena_size_bytes = default_arena_size_in_bytes, float fpu_reduction = 0.0f,
         std::shared_ptr<StateEncoder> encoder = nullptr);

    std::pair<std::vector<float>, float> search(const Game &game, int num_simulations,
                                                int batch_size);

    // Gumbel AlphaZero (Danihelka, Guez, van Hasselt & Silver, "Policy improvement
    // by planning with Gumbel", ICLR 2022): replaces search()'s Dirichlet-noised,
    // pure-PUCT root action selection with Gumbel-Top-k sampling of
    // max_num_considered_actions candidates followed by sequential halving of the
    // simulation budget across them. This gives a policy-improvement guarantee even
    // at small simulation counts, where plain PUCT can fail to move enough visits
    // off a bad prior. Only the *root's* action choice changes - every simulation
    // still descends the rest of the tree with ordinary PUCT (search_gumbel's own
    // c_init/c_base members are reused for that). Kept side by side with search()
    // rather than replacing it, so existing callers/tuning are unaffected.
    //
    // max_num_considered_actions: size of the initial Gumbel-Top-k candidate set
    // (m in the paper); c_visit/c_scale: the sigma() transform's constants that
    // turn a completed Q-value into a score comparable to raw policy logits -
    // 50/1.0 are the paper's board-game (Go/chess/shogi) defaults.
    //
    // Unlike search(), the action to *play* is returned explicitly
    // (chosen_action) rather than left for the caller to derive from pi: the
    // paper's policy-improvement guarantee for the played move requires
    // argmax(g + logit + sigma(completedQ)) over the sequential-halving
    // survivors - the Gumbel term g is deliberately absent from pi (see the
    // improved-policy comment in the implementation), so argmax(pi) is *not*
    // the same action and doesn't carry the guarantee.
    struct gumbel_result {
        std::vector<float> pi;
        float root_value{};
        // The sequential-halving winner; -1 only when the root is terminal
        // (no legal actions to choose from).
        int chosen_action = -1;
    };
    gumbel_result search_gumbel(const Game &game, int num_simulations, int batch_size,
                                int max_num_considered_actions = 16, float c_visit = 50.0f,
                                float c_scale = 1.0f);

  private:
    class Node;
    void evaluate_batch(std::vector<std::pair<Node *, std::shared_ptr<Game>>> &leaves,
                        std::pmr::memory_resource *pool);

    std::vector<std::pair<int, float>> get_policy_from_logits(const inference_result &res,
                                                              bool dirichletNoise = false) const;

    static std::vector<float> sample_dirichlet(const std::vector<float> &alpha);

    // Gumbel(0,1) i.i.d. draws, one per legal action - used both for the initial
    // Gumbel-Top-k candidate selection and, added to logits again at every
    // sequential-halving cut, to keep candidate rankings consistent with a single
    // set of noise draws throughout search_gumbel() (this is what makes
    // Gumbel-Top-k equivalent to sampling without replacement from softmax(logits),
    // rather than an independent re-sample at each phase).
    static std::vector<float> sample_gumbel(size_t n);
};

#endif // MCTS_HPP
