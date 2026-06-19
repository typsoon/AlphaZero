#include "self_play.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::string & std::vector support

// Assuming you have these headers somewhere

namespace py = pybind11;

PYBIND11_MODULE(self_play_bind, m) {

    m.def("self_play", &self_play, py::arg("game"), py::arg("network_path"), py::arg("replay_buf"),
          py::arg("num_games") = 100, py::arg("thread_count") = 1);
}
