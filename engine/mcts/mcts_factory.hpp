#ifndef MCTSFACTORY_HPP
#define MCTSFACTORY_HPP

#include "mcts.hpp"
#include <inferer.hpp>
#include <memory>

class MCTSFactory {
  private:
    std::reference_wrapper<InfererFactory> inferer_factory;
    float c_init;
    float c_base;
    float eps;
    float alpha;
    size_t arena_size_bytes;

  public:
    MCTSFactory(InfererFactory &inferer_factory, float c_init = 1.25, float c_base = 19652,
                float eps = 0.25, float alpha = 0.3,
                size_t arena_size_bytes = default_arena_size_in_bytes);

    std::unique_ptr<MCTS> get_mcts();
};

#endif // MCTSFACTORY_HPP
