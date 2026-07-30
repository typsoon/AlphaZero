#include "inference_server_args.hpp"

#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

namespace {

std::string generate_uuid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    return ss.str();
}

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
              << "  --game <game>           Game name for default socket path (default: "
                 "connect4)\n"
              << "  --socket <path>         Unix socket path (default: "
                 "/tmp/alphazero-inference/<game>/<network_name>/<uuid>.sock)\n"
              << "  --mcts-search-depth <depth> MCTS search depth (default: 800)\n"
              << "  --chess-encoder-history <N> Chess only: 0 (default) = 19-plane "
                 "ChessEncoderV1; 1/4/8 = ChessEncoderV2History(N) for a history-encoder net\n"
              << "  --use-gumbel-search     Use search_gumbel() (Gumbel-Top-k root "
                 "sampling) instead of plain-PUCT search() (default: off)\n"
              << "  --max-num-considered-actions <N> Gumbel-Top-k candidate set size, "
                 "only used with --use-gumbel-search (default: 16)\n"
              << "  --full-search-probability <p> Fraction of requests using the full "
                 "mcts-search-depth; the rest use --fast-mcts-simulations (default: 1.0, "
                 "i.e. always full)\n"
              << "  --fast-mcts-simulations <N> Simulation count for the \"fast\" branch "
                 "above, only used when --full-search-probability < 1.0\n"
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
        if (read_option_value(i, argc, argv, arg, "--game", args.game, error)) {
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
        std::string enc_hist_str;
        if (read_option_value(i, argc, argv, arg, "--chess-encoder-history", enc_hist_str, error)) {
            try {
                args.chess_encoder_history = std::stoi(enc_hist_str);
            } catch (const std::exception &) {
                error = "Invalid value for --chess-encoder-history: " + enc_hist_str;
                return false;
            }
            if (args.chess_encoder_history != 0 && args.chess_encoder_history != 1 &&
                args.chess_encoder_history != 4 && args.chess_encoder_history != 8) {
                error = "--chess-encoder-history must be 0 (default 19-plane), 1, 4, or 8";
                return false;
            }
            continue;
        }
        if (arg == "--use-gumbel-search") {
            args.use_gumbel_search = true;
            continue;
        }
        std::string max_considered_str;
        if (read_option_value(i, argc, argv, arg, "--max-num-considered-actions",
                              max_considered_str, error)) {
            try {
                args.max_num_considered_actions = std::stoi(max_considered_str);
            } catch (const std::exception &) {
                error = "Invalid value for --max-num-considered-actions: " + max_considered_str;
                return false;
            }
            continue;
        }
        std::string full_search_prob_str;
        if (read_option_value(i, argc, argv, arg, "--full-search-probability",
                              full_search_prob_str, error)) {
            try {
                args.full_search_probability = std::stof(full_search_prob_str);
            } catch (const std::exception &) {
                error = "Invalid value for --full-search-probability: " + full_search_prob_str;
                return false;
            }
            if (args.full_search_probability < 0.0f || args.full_search_probability > 1.0f) {
                error = "--full-search-probability must be between 0.0 and 1.0";
                return false;
            }
            continue;
        }
        std::string fast_sims_str;
        if (read_option_value(i, argc, argv, arg, "--fast-mcts-simulations", fast_sims_str,
                              error)) {
            try {
                args.fast_mcts_simulations = std::stoi(fast_sims_str);
            } catch (const std::exception &) {
                error = "Invalid value for --fast-mcts-simulations: " + fast_sims_str;
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

    if (args.socket.empty()) {
        std::filesystem::path net_path(args.network_path);
        std::string network_name = net_path.stem().string();
        args.socket = "/tmp/alphazero-inference/" + args.game + "/" + network_name + "/" +
                      generate_uuid() + ".sock";
    }

    return true;
}
