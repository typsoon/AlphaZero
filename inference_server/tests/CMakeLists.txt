find_package(CppUTest CONFIG REQUIRED)

add_executable(
  inference_server_tests
  test_main.cpp ../model_wrapper/model_wrapper.cpp
  ../schema_validator/schema_validator.cpp
  ../args_parser/inference_server_args.cpp)

target_link_libraries(
  inference_server_tests
  PRIVATE CppUTest
          nlohmann_json::nlohmann_json
          nlohmann_json_schema_validator::validator
          game
          inference
          mcts
          "${TORCH_LIBRARIES}")

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  target_link_libraries(inference_server_tests PRIVATE CUDA::cupti)
endif()

target_include_directories(
  inference_server_tests PRIVATE ${TORCH_INCLUDE_DIRS}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_compile_definitions(
  inference_server_tests
  PRIVATE ALPHAZERO_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../.."
          CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)

# Add test target
add_test(NAME inference_server_tests COMMAND inference_server_tests)
