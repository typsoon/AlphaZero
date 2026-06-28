#include "game/game.hpp"
#include "mcts.hpp"
#include "replay_buffer.hpp"
#include <c10/core/Device.h>
#include <game/chess.hpp>
#include <game/connect4.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
// #include <torch/script.h> // For LibTorch (includes pybind11)
#include <torch/torch.h>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::unique_ptr<T>)

PYBIND11_MODULE(engine_bind, m) {
    try {
        py::class_<Transition>(m, "Transition")
            .def(py::init<torch::Tensor, torch::Tensor, float>())
            .def_readwrite("state", &Transition::state)
            .def_readwrite("policy", &Transition::policy)
            .def_readwrite("reward", &Transition::reward);

        py::class_<ReplayBuffer>(m, "ReplayBuffer")
            .def(py::init<size_t>())
            .def("add", &ReplayBuffer::add)
            .def("sample", &ReplayBuffer::sample)
            .def("get_size", &ReplayBuffer::get_size);

        py::class_<Game, std::shared_ptr<Game>>(m, "Game")
            .def("get_legal_actions", &Game::get_legal_actions)
            .def("step", &Game::step)
            .def("reset", &Game::reset)
            .def_property_readonly("is_terminal", &Game::is_terminal)
            .def_property_readonly("current_player", &Game::get_current_player);

        py::class_<Game2D<6, 7>, Game, std::shared_ptr<Game2D<6, 7>>>(m, "Game2D_6_7")
            .def("get_board_state", &Game2D<6, 7>::get_board_state)
            .def_readonly_static("ROWS", &Game2D<6, 7>::ROWS)
            .def_readonly_static("COLS", &Game2D<6, 7>::COLS);

        py::class_<Connect4, Game2D<6, 7>, std::shared_ptr<Connect4>>(m, "Connect4")
            .def(py::init<>())
            .def(py::init<const std::vector<std::vector<int>> &>(), py::arg("initial_board"))
            .def_readonly_static("action_dim", &Connect4::action_dim)
            .def_property_readonly_static("state_dim", [](py::object /* self */) {
                return Connect4::state_dim; // or use Connect4::state_dim
            });
        // .def("reset", &Connect4::reset)
        // .def("getActionSize", &Connect4::getActionSize)
        // .def("getLegalActions", &Connect4::getLegalActions)
        // .def("step", &Connect4::step)
        // .def("is_terminal", &Connect4::is_terminal)
        // .def("reward", &Connect4::reward)
        // .def("get_canonical_state", &Connect4::get_canonical_state)
        // .def("clone", &Connect4::clone)
        // .def("render", &Connect4::render)
        ;

        py::class_<Game2D<8, 8>, Game, std::shared_ptr<Game2D<8, 8>>>(m, "Game2D_8_8")
            .def("get_board_state", &Game2D<8, 8>::get_board_state)
            .def_readonly_static("ROWS", &Game2D<8, 8>::ROWS)
            .def_readonly_static("COLS", &Game2D<8, 8>::COLS);

        py::class_<Chess, Game2D<8, 8>, std::shared_ptr<Chess>>(m, "Chess")
            .def(py::init<>())
            .def_readonly_static("action_dim", &Chess::action_dim)
            .def_property_readonly_static("state_dim",
                                          [](py::object /* self */) { return Chess::state_dim; });

        py::class_<MCTS>(m, "MCTS")
            .def(py::init<std::string, torch::Device, float, float, float, float>(),
                 py::arg("network_path"), py::arg("device"), py::arg("c_init") = 1.25f,
                 py::arg("c_base") = 19652.0f, py::arg("eps") = 0.25f, py::arg("alpha") = 0.3f)
            .def("search", &MCTS::search, py::arg("game"), py::arg("num_simulations") = 800,
                 py::arg("batch_size") = 32);

    } catch (const std::exception &e) {
        py::print("Exception during binding:", e.what());
        throw;
    }
}
