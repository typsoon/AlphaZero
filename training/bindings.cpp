#include "self_play.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::string & std::vector support

// Assuming you have these headers somewhere

namespace py = pybind11;

PYBIND11_MODULE(self_play_bind, m) {

    m.def("self_play", &self_play, py::arg("game"), py::arg("network_path"), py::arg("replay_buf"),
          py::arg("num_games") = 100, py::arg("thread_count") = 1,
          py::arg("mcts_num_simulations") = 800, py::arg("mcts_batch_size") = 32,
          py::arg("max_moves") = 512, py::arg("fast_mcts_num_simulations") = 100,
          py::arg("full_search_probability") = 0.25f,
          py::arg("transposition_cache_entries") = 1000000, py::arg("use_gumbel_search") = false,
          py::arg("max_num_considered_actions") = 16, py::arg("resignation_enabled") = false,
          py::arg("resignation_threshold") = -0.95f, py::arg("resignation_consecutive_moves") = 3,
          py::arg("resignation_min_ply") = 60, py::arg("resignation_disable_probability") = 0.1f,
          py::arg("fpu_reduction") = 0.0f, py::arg("encoder") = std::shared_ptr<StateEncoder>(),
          py::arg("self_play_encoder") = std::shared_ptr<StateEncoder>(),
          py::arg("value_network_path") = std::string(""),
          py::arg("value_network_encoder") = std::shared_ptr<StateEncoder>(),
          py::arg("dirichlet_epsilon") = 0.25f);

    m.def("self_play_connect4", &self_play, py::arg("game"), py::arg("network_path"),
          py::arg("replay_buf"), py::arg("num_games") = 100, py::arg("thread_count") = 1,
          py::arg("mcts_num_simulations") = 800, py::arg("mcts_batch_size") = 32,
          py::arg("max_moves") = 512, py::arg("fast_mcts_num_simulations") = 100,
          py::arg("full_search_probability") = 0.25f,
          py::arg("transposition_cache_entries") = 1000000, py::arg("use_gumbel_search") = false,
          py::arg("max_num_considered_actions") = 16, py::arg("resignation_enabled") = false,
          py::arg("resignation_threshold") = -0.95f, py::arg("resignation_consecutive_moves") = 3,
          py::arg("resignation_min_ply") = 60, py::arg("resignation_disable_probability") = 0.1f,
          py::arg("fpu_reduction") = 0.0f, py::arg("encoder") = std::shared_ptr<StateEncoder>(),
          py::arg("self_play_encoder") = std::shared_ptr<StateEncoder>(),
          py::arg("value_network_path") = std::string(""),
          py::arg("value_network_encoder") = std::shared_ptr<StateEncoder>(),
          py::arg("dirichlet_epsilon") = 0.25f);
}
