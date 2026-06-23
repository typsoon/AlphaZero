#include "inference_server_args.hpp"

#include <iostream>
#include <string>

namespace {

bool read_option_value(int &index, int argc, char *argv[], const std::string &arg,
                       const std::string &long_name, std::string &out_value, std::string &error) {
    const std::string prefix = long_name + "=";
    if (arg.rfind(prefix, 0) == 0) {
        out_value = arg.substr(prefix.size());
        if (out_value.empty()) {
            error = "Missing value for " + long_name;
            return false;
        }
        return true;
    }

    if (arg == long_name) {
        if (index + 1 >= argc) {
            error = "Missing value for " + long_name;
            return false;
        }
        ++index;
        out_value = argv[index];
        return true;
    }

    return false;
}

} // namespace

void print_inference_server_usage(const char *program_name) {
    std::cerr << "Usage: " << program_name
              << " --network-path <path> [--device <cuda|cpu>] [--socket <path>]\n"
              << "\n"
              << "Arguments:\n"
              << "  --network-path <path>   Path to the trained network file "
                 "(required)\n"
              << "  --device <device>       Device to run inference on (default: "
                 "cuda)\n"
              << "  --socket <path>         Unix socket path (default: "
                 "/tmp/alphazero.sock)\n"
              << "  --mcts-search-depth <depth> MCTS search depth (default: 800)\n"
              << "  -h, --help              Show this help message\n";
}

bool parse_inference_server_args(int argc, char *argv[], InferenceServerArgs &args, bool &show_help,
                                 std::string &error) {
    show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            show_help = true;
            return true;
        }

        if (read_option_value(i, argc, argv, arg, "--network-path", args.network_path, error)) {
            continue;
        }
        if (read_option_value(i, argc, argv, arg, "--device", args.device, error)) {
            continue;
        }
        if (read_option_value(i, argc, argv, arg, "--socket", args.socket, error)) {
            continue;
        }
        std::string depth_str;
        if (read_option_value(i, argc, argv, arg, "--mcts-search-depth", depth_str, error)) {
            try {
                args.mcts_search_depth = std::stoi(depth_str);
            } catch (const std::exception &) {
                error = "Invalid value for --mcts-search-depth: " + depth_str;
                return false;
            }
            continue;
        }
        std::string batch_str;
        if (read_option_value(i, argc, argv, arg, "--mcts-batch-size", batch_str, error)) {
            try {
                args.mcts_batch_size = std::stoi(batch_str);
            } catch (const std::exception &) {
                error = "Invalid value for --mcts-batch-size: " + batch_str;
                return false;
            }
            continue;
        }

        error = "Unknown argument: " + arg;
        return false;
    }

    if (args.network_path.empty()) {
        error = "Missing required argument: --network-path";
        return false;
    }

    return true;
}
