#include "./server.hpp"
#include "crow/app.h"
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace {

class SocketPathGuard {
  std::string path;

public:
  explicit SocketPathGuard(std::string path) : path(std::move(path)) {
    std::error_code ec;
    std::filesystem::remove(this->path, ec);
  }

  ~SocketPathGuard() noexcept {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

} // namespace

template <typename M>
void set_up_and_run_server(std::string socket_path, std::shared_ptr<M> model) {
  SocketPathGuard socket_path_guard(socket_path);
  crow::SimpleApp app;

  CROW_ROUTE(app, "/predict")([model](auto &req) {
    (void)req;
    (void)model;
    return "Hello";
  });

  app.local_socket_path(socket_path).run();
}

template void
    set_up_and_run_server<ModelWrapper>(std::string,
                                        std::shared_ptr<ModelWrapper>);
