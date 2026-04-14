#ifndef ALPHAZERO_INFERENCE_SERVER_SERVER_SERVER_HPP
#define ALPHAZERO_INFERENCE_SERVER_SERVER_SERVER_HPP

#include <memory>
#include <string>

template <typename M>
void set_up_and_run_server(std::string address, std::shared_ptr<M> model);

#endif // ALPHAZERO_INFERENCE_SERVER_SERVER_SERVER_HPP
