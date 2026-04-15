#ifndef ALPHAZERO_INFERENCE_SERVER_SERVER_SERVER_HPP
#define ALPHAZERO_INFERENCE_SERVER_SERVER_SERVER_HPP

#include <memory>
#include <string>

template <typename M, typename S>
void set_up_and_run_server(std::string address, std::shared_ptr<M> model,
                           std::shared_ptr<S> schema_validator);

#endif // ALPHAZERO_INFERENCE_SERVER_SERVER_SERVER_HPP
