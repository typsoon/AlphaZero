#ifndef ALPHAZERO_INFERENCE_SERVER_MODEL_WRAPPER_MODEL_WRAPPER_HPP
#define ALPHAZERO_INFERENCE_SERVER_MODEL_WRAPPER_MODEL_WRAPPER_HPP

#include <memory>
#include <string>

#include <vector>

class ModelWrapper {
  public:
    virtual std::string encode_payload(const std::vector<float> &policy, float value) = 0;
    virtual ~ModelWrapper() = default;
    virtual std::string predict(const std::string &request_payload) = 0;
};

// use_gumbel_search/max_num_considered_actions/full_search_probability/
// fast_mcts_simulations mirror training_params/*.json's self-play search
// config (see InferenceServerArgs) - defaults reproduce this function's
// original behavior (plain PUCT, always the full mcts_search_depth).
std::shared_ptr<ModelWrapper> create_connect4_model_wrapper(
    const std::string &network_path, const std::string &device, int mcts_search_depth,
    int mcts_batch_size, bool use_gumbel_search = false, int max_num_considered_actions = 16,
    float full_search_probability = 1.0f, int fast_mcts_simulations = 0,
    float dirichlet_epsilon = 0.25f, float fpu_reduction = 0.0f);

// chess_encoder_history: 0 = default 19-plane ChessEncoderV1; N in {1,4,8} =
// ChessEncoderV2History(N) for a history-encoder net. chess_encoder_flip_white
// selects ChessEncoderV2History's row-flip convention (see its class comment)
// - only ever true for a checkpoint transplanted from engine-zoo. The
// remaining trailing params match create_connect4_model_wrapper's above.
std::shared_ptr<ModelWrapper> create_chess_model_wrapper(
    const std::string &network_path, const std::string &device, int mcts_search_depth,
    int mcts_batch_size, int chess_encoder_history = 0, bool use_gumbel_search = false,
    int max_num_considered_actions = 16, float full_search_probability = 1.0f,
    int fast_mcts_simulations = 0, float dirichlet_epsilon = 0.25f,
    bool chess_encoder_flip_white = false, float fpu_reduction = 0.0f);

#endif // ALPHAZERO_INFERENCE_SERVER_MODEL_WRAPPER_MODEL_WRAPPER_HPP
