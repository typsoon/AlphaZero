#ifndef ALPHAZERO_INFERENCE_SERVER_ARGS_PARSER_INFERENCE_SERVER_ARGS_HPP
#define ALPHAZERO_INFERENCE_SERVER_ARGS_PARSER_INFERENCE_SERVER_ARGS_HPP

#include <string>

struct InferenceServerArgs {
  std::string network_path;
  std::string device = "cuda";
  std::string socket = "/tmp/alphazero.sock";
};

void print_inference_server_usage(const char *program_name);

bool parse_inference_server_args(int argc, char *argv[], InferenceServerArgs &args,
                                 bool &show_help, std::string &error);

#endif // ALPHAZERO_INFERENCE_SERVER_ARGS_PARSER_INFERENCE_SERVER_ARGS_HPP
