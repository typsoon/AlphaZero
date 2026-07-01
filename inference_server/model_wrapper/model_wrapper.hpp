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

std::shared_ptr<ModelWrapper> create_connect4_model_wrapper(const std::string &network_path,
                                                            const std::string &device,
                                                            int mcts_search_depth,
                                                            int mcts_batch_size);

std::shared_ptr<ModelWrapper> create_chess_model_wrapper(const std::string &network_path,
                                                         const std::string &device,
                                                         int mcts_search_depth,
                                                         int mcts_batch_size);

#endif // ALPHAZERO_INFERENCE_SERVER_MODEL_WRAPPER_MODEL_WRAPPER_HPP
