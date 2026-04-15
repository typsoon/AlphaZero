#ifndef ALPHAZERO_INFERENCE_SERVER_MODEL_WRAPPER_MODEL_WRAPPER_HPP
#define ALPHAZERO_INFERENCE_SERVER_MODEL_WRAPPER_MODEL_WRAPPER_HPP

#include <memory>
#include <string>

class ModelWrapper {
public:
  virtual ~ModelWrapper() = default;
  virtual std::string predict(const std::string &request_payload) = 0;
};

std::shared_ptr<ModelWrapper>
create_connect4_model_wrapper(const std::string &network_path,
                              const std::string &device);

#endif // ALPHAZERO_INFERENCE_SERVER_MODEL_WRAPPER_MODEL_WRAPPER_HPP
