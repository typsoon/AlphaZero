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

add_library(engine STATIC "" test_libtorch.cpp)
target_link_libraries(engine
    PUBLIC
        game
        inference
        mcts
        utils
        "${TORCH_LIBRARIES}"
)

target_include_directories(engine PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_libtorch test_libtorch.cpp)
target_link_libraries(test_libtorch
  "${TORCH_LIBRARIES}"
)

#  Pybind module
add_library(engine_bind MODULE bindings.cpp)
set_target_properties(engine_bind PROPERTIES PREFIX "")

find_library(TORCH_PYTHON_LIBRARY torch_python
PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(engine_bind
    PRIVATE
        engine
        "${TORCH_PYTHON_LIBRARY}"
        "${TORCH_LIBRARIES}"
)

target_include_directories(engine_bind PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
