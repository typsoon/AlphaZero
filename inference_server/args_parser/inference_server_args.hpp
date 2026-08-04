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
    // ChessEncoderV2History's row-flip convention: false (default) = this
    // engine's own choice (Black flipped); true = the engine-zoo reference's
    // own convention (White flipped). Only ever needed for a checkpoint whose
    // weights were transplanted from engine-zoo (e.g. via
    // convert_safetensors_v2.py) - never for a network this repo trained
    // itself. Ignored when chess_encoder_history is 0.
    bool chess_encoder_flip_white{false};
    // The remaining fields mirror training_params/*.json's self-play search
    // config (see AlphaZeroTrainer's self_play_and_train_loop), letting the
    // server reproduce the exact search behavior training data was generated
    // with, rather than always using the plain-PUCT search() path regardless
    // of what a checkpoint was actually trained under. Defaults preserve the
    // server's original behavior (plain PUCT, always the full
    // mcts_search_depth) when none of these are passed.
    bool use_gumbel_search{false};
    // search_gumbel's Gumbel-Top-k candidate set size (m in the paper) - only
    // consulted when use_gumbel_search is set.
    int max_num_considered_actions{16};
    // Fraction of requests that get the full mcts_search_depth; the rest use
    // fast_mcts_simulations instead. 1.0 (default) = always full search, the
    // sensible default for serving real moves - self-play uses this to make
    // most training games cheaper, which isn't a reason to shortchange an
    // actual request unless explicitly asked to reproduce that behavior.
    float full_search_probability{1.0f};
    // Simulation count used for the "fast" branch above. Only meaningful when
    // full_search_probability < 1.0; 0 (default, i.e. unset) falls back to
    // mcts_search_depth so a stray fast_mcts_simulations=0 can't silently
    // zero out the search.
    int fast_mcts_simulations{0};
    // Weight of Dirichlet root noise mixed into plain-PUCT search()'s root
    // policy: policy = (1-eps)*policy + eps*noise. Matches MCTS's own default
    // (0.25, AlphaGo Zero's self-play value). search_gumbel() never uses this
    // (Gumbel-Top-k IS its root exploration mechanism). 0.0 disables root
    // noise entirely - the setting an arena or a "give me this net's actual
    // best move" server wants, since noise is a self-play exploration device,
    // not something serving/evaluation should have on by default.
    float dirichlet_epsilon{0.25f};
    // First-play-urgency reduction (see mcts.hpp's fpu_reduction comment): an
    // unvisited MCTS child is scored at its parent's running value minus this
    // amount instead of the assume-draw 0.0. 0.0 (default) preserves the
    // server's original behavior; 0.33 matches some reference PUCT
    // implementations (e.g. engine-zoo).
    float fpu_reduction{0.0f};
};

void print_inference_server_usage(const char *program_name);

bool parse_inference_server_args(int argc, char *argv[], InferenceServerArgs &args, bool &show_help,
                                 std::string &error);

#endif // ALPHAZERO_INFERENCE_SERVER_ARGS_PARSER_INFERENCE_SERVER_ARGS_HPP
