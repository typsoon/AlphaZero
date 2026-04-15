cmake_minimum_required(VERSION 3.15)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(inference_server LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(Torch REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(nlohmann_json_schema_validator CONFIG REQUIRED)
find_package(Crow CONFIG REQUIRED)

if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
  set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "")
endif()

if(NOT TARGET game)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../engine/game
                   ${CMAKE_CURRENT_BINARY_DIR}/engine_game)
endif()

if(NOT TARGET inference)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../engine/inference
                   ${CMAKE_CURRENT_BINARY_DIR}/engine_inference)
endif()

if(NOT TARGET mcts)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../engine/mcts
                   ${CMAKE_CURRENT_BINARY_DIR}/engine_mcts)
endif()

# Link it to your executable
add_executable(inference_server
    inference_server.cpp
    args_parser/inference_server_args.cpp
    server/server.cpp
    model_wrapper/model_wrapper.cpp
    schema_validator/schema_validator.cpp
)
target_link_libraries(inference_server PRIVATE Crow::Crow
     nlohmann_json::nlohmann_json
      nlohmann_json_schema_validator::validator
      game
      inference
      mcts
      "${TORCH_LIBRARIES}"
 )

target_include_directories(inference_server PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_compile_definitions(inference_server PRIVATE
    ALPHAZERO_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
