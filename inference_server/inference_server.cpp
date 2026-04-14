#include "args_parser/inference_server_args.hpp"
#include "model_wrapper/connect4_model_wrapper.hpp"
#include "model_wrapper/model_wrapper.hpp"
#include "server/server.hpp"
#include <iostream>

namespace {

constexpr int PORT = 4567;

int run_server(const InferenceServerArgs &args) {
  // TODO: replace with inference server startup once C++ service is wired.
  set_up_and_run_server<ModelWrapper>(args.socket,
                                      get_connect4_model_wrapper());
  std::cout << "network_path=" << args.network_path << '\n'
            << "device=" << args.device << '\n'
            << "socket=" << args.socket << '\n';
  return 0;
}

} // namespace

int main(int argc, char *argv[]) {
  InferenceServerArgs args;
  bool show_help = false;
  std::string error;

  if (!parse_inference_server_args(argc, argv, args, show_help, error)) {
    std::cerr << error << '\n';
    print_inference_server_usage(argv[0]);
    return 2;
  }

  if (show_help) {
    print_inference_server_usage(argv[0]);
    return 0;
  }

  return run_server(args);
}
