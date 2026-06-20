cmake_minimum_required(VERSION 3.18)
project(self_play_bind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

add_library(self_play_bind MODULE bindings.cpp self_play.cpp)
set_target_properties(self_play_bind PROPERTIES PREFIX "")

target_include_directories(self_play_bind PRIVATE ${Python3_INCLUDE_DIRS}
                                                  ${TORCH_INCLUDE_DIRS})

target_link_libraries(self_play_bind PRIVATE engine spdlog::spdlog
                                             "${TORCH_LIBRARIES}")

# set_target_properties(self_play_bind PROPERTIES INSTALL_RPATH "$ORIGIN" # Look
# in the same dir as the .so file BUILD_WITH_INSTALL_RPATH TRUE     # Use RPATH
# even before installation )
