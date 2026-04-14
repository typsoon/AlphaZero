cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(nlohmann_json_schema_validator CONFIG REQUIRED)
find_package(Crow CONFIG REQUIRED)

# Link it to your executable
add_executable(inference_server
    inference_server.cpp
    args_parser/inference_server_args.cpp
    server/server.cpp
    model_wrapper/model_wrapper.cpp
    model_wrapper/connect4_model_wrapper.cpp
)
target_link_libraries(inference_server PRIVATE Crow::Crow
     nlohmann_json::nlohmann_json
     nlohmann_json_schema_validator::validator
)
