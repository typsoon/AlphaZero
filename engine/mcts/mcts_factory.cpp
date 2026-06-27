#include "mcts_factory.hpp"

MCTSFactory::MCTSFactory(InfererFactory &inferer_factory, float c_init, float c_base, float eps,
                         float alpha)
    : inferer_factory(inferer_factory), c_init(c_init), c_base(c_base), eps(eps), alpha(alpha) {}

std::unique_ptr<MCTS> MCTSFactory::get_mcts() {
    return std::make_unique<MCTS>(inferer_factory.get().get_inferer(), c_init, c_base, eps, alpha);
}
