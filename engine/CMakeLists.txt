cmake_minimum_required(VERSION 3.16)
project(engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
# find_package(pybind11 REQUIRED CONFIG)

include_directories(${Python3_INCLUDE_DIRS})

add_subdirectory(game)
add_subdirectory(mcts)
add_subdirectory(inference)
add_subdirectory(utils)
if(BUILD_TESTS OR BUILD_PROFILING)
  add_subdirectory(profiling)
endif()

add_library(engine INTERFACE)
target_link_libraries(engine INTERFACE game inference mcts utils spdlog::spdlog
                                       "${TORCH_LIBRARIES}")

target_include_directories(engine INTERFACE ${TORCH_INCLUDE_DIRS}
                                            ${CMAKE_CURRENT_SOURCE_DIR})

# Pybind module
add_library(engine_bind MODULE bindings.cpp)
set_target_properties(engine_bind PROPERTIES PREFIX "")

find_library(TORCH_PYTHON_LIBRARY torch_python PATH
             "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(engine_bind PRIVATE engine "${TORCH_PYTHON_LIBRARY}"
                                          "${TORCH_LIBRARIES}")

target_include_directories(engine_bind PRIVATE ${Python3_INCLUDE_DIRS}
                                               ${TORCH_INCLUDE_DIRS})

# Tests
if(BUILD_TESTS)
  find_package(CppUTest CONFIG REQUIRED)
  add_executable(test_chess tests/test_chess.cpp)
  target_link_libraries(test_chess PRIVATE engine CppUTest "${TORCH_LIBRARIES}")
  target_compile_definitions(test_chess
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)

  add_executable(test_mcts tests/test_mcts.cpp)
  target_link_libraries(test_mcts PRIVATE engine CppUTest "${TORCH_LIBRARIES}")
  target_compile_definitions(test_mcts
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)
endif()
