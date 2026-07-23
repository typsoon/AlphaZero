#ifndef ALPHAZERO_INFERENCE_SERVER_ARGS_PARSER_INFERENCE_SERVER_ARGS_HPP
#define ALPHAZERO_INFERENCE_SERVER_ARGS_PARSER_INFERENCE_SERVER_ARGS_HPP

#include <string>

struct InferenceServerArgs {
    std::string network_path;
    std::string device = "cuda";
    std::string socket = "";
    std::string game = "connect4";
    int mcts_search_depth{800};
    int mcts_batch_size{32};
    // Chess only: 0 = default 19-plane ChessEncoderV1; N in {1,4,8} =
    // ChessEncoderV2History(N) for a history-encoder net.
    int chess_encoder_history{0};
};

void print_inference_server_usage(const char *program_name);

bool parse_inference_server_args(int argc, char *argv[], InferenceServerArgs &args, bool &show_help,
                                 std::string &error);

#endif // ALPHAZERO_INFERENCE_SERVER_ARGS_PARSER_INFERENCE_SERVER_ARGS_HPP
