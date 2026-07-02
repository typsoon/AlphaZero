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

// Sized for chess, the larger of the two games: Node::expand() allocates a
// children array sized to the full action space (20480 * 8 bytes = ~160KB) per
// expansion, so at the default 800 simulations/search a single search() call can
// need up to ~130MB. The arena is fully reset (pool.release()) at the start of
// every search() call, so undersizing it means most expansions overflow into the
// upstream allocator (malloc) on effectively every move.
constexpr size_t default_arena_size_in_bytes = static_cast<const size_t>(256 * 1024 * 1024);

// TODO: test as many methods as you can
class MCTS {
    using InfererPtr = unique_ptr<Inferer>;
    InfererPtr network;
    float c_init;
    float c_base;
    float eps;
    float alpha;
    torch::Device device;

    std::vector<std::byte> arena_buffer;
    std::pmr::monotonic_buffer_resource pool;

  public:
    MCTS(unique_ptr<Inferer> &&network, float c_init = 1.25f, float c_base = 19652.0f,
         float eps = 0.25f, float alpha = 0.3f,
         size_t arena_size_bytes = default_arena_size_in_bytes);

    MCTS(std::string network_path, torch::Device device, float c_init = 1.25f,
         float c_base = 19652.0f, float eps = 0.25f, float alpha = 0.3f,
         size_t arena_size_bytes = default_arena_size_in_bytes);

    std::pair<std::vector<float>, float> search(const Game &game, int num_simulations,
                                                int batch_size);

  private:
    class Node;
    void evaluate_batch(std::vector<std::pair<Node *, std::shared_ptr<Game>>> &leaves,
                        std::pmr::memory_resource *pool);

    std::vector<std::pair<int, float>> get_policy_from_logits(const inference_result &res,
                                                              bool dirichletNoise = false) const;

    static std::vector<float> sample_dirichlet(const std::vector<float> &alpha);
};

#endif // MCTS_HPP
