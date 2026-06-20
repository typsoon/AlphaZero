#include "args_parser/inference_server_args.hpp"
#include "model_wrapper/model_wrapper.hpp"
#include "schema_validator/schema_validator.hpp"
#include "server/server.hpp"
#include <filesystem>
#include <spdlog/spdlog.h>

namespace {

int run_server(const InferenceServerArgs &args) {
    auto schema_path = std::filesystem::path(ALPHAZERO_REPO_ROOT) / "game_states" / "connect4.json";

    set_up_and_run_server<ModelWrapper, SchemaValidator>(
        args.socket, create_connect4_model_wrapper(args.network_path, args.device),
        get_schema_validator(schema_path));
    spdlog::info("network_path={}\ndevice={}\nsocket={}", args.network_path, args.device,
                 args.socket);
    return 0;
}

} // namespace

int main(int argc, char *argv[]) {
    InferenceServerArgs args;
    bool show_help = false;
    std::string error;

    if (!parse_inference_server_args(argc, argv, args, show_help, error)) {
        spdlog::error(error);
        print_inference_server_usage(argv[0]);
        return 2;
    }

    if (show_help) {
        print_inference_server_usage(argv[0]);
        return 0;
    }

    try {
        return run_server(args);
    } catch (const std::exception &e) {
        spdlog::error("Server terminated due to error: {}", e.what());
        return 1;
    }
}
