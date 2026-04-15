#include "./server.hpp"
#include "../model_wrapper/model_wrapper.hpp"
#include "../schema_validator/schema_validator.hpp"
#include "crow/app.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <iostream>
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

template <typename M, typename S>
void set_up_and_run_server(std::string socket_path, std::shared_ptr<M> model,
                           std::shared_ptr<S> schema_validator) {
  SocketPathGuard socket_path_guard(socket_path);
  crow::SimpleApp app;

  CROW_ROUTE(app, "/predict")
      .methods(crow::HTTPMethod::Post)([model, schema_validator](
                                           const crow::request &req,
                                           crow::response &resp) {
        try {
          const auto request_json = nlohmann::json::parse(req.body);
          if (!request_json.is_object() ||
              !request_json.contains("game_state")) {
            resp.code = 400;
            resp.write(R"({"error":"Missing game_state in request"})");
            resp.end();
            return;
          }

          const auto &game_state = request_json.at("game_state");
          if (!game_state.is_object()) {
            resp.code = 400;
            resp.write(R"({"error":"game_state must be an object"})");
            resp.end();
            return;
          }

          if (!schema_validator->is_a_valid_boardstate(game_state.dump())) {
            resp.code = 400;
            resp.write(R"({"error":"game_state does not match schema"})");
            resp.end();
            return;
          }

          const auto model_response = model->predict(game_state.dump());
          resp.code = 200;
          resp.set_header("Content-Type", "application/json");
          resp.write(model_response);
          resp.end();
          return;
        } catch (const nlohmann::json::exception &) {
          resp.code = 400;
          resp.write(R"({"error":"Malformed JSON request"})");
          resp.end();
          return;
        } catch (const std::exception &) {
          resp.code = 500;
          resp.write(R"({"error":"Internal server error"})");
          resp.end();
          return;
        }
      });

  CROW_ROUTE(app, "/health")
      .methods(crow::HTTPMethod::Get)([](const crow::request &,
                                         crow::response &resp) {
    resp.code = 200;
    resp.set_header("Content-Type", "application/json");
    resp.write(R"({"status":"healthy"})");
    resp.end();
  });

  app.local_socket_path(socket_path).run();
}

template void
    set_up_and_run_server<ModelWrapper, SchemaValidator>(
        std::string, std::shared_ptr<ModelWrapper>,
        std::shared_ptr<SchemaValidator>);
