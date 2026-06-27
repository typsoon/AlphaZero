add_executable(run_self_play EXCLUDE_FROM_ALL run_self_play.cpp
                                              ../../training/self_play.cpp)

target_link_libraries(run_self_play PRIVATE engine spdlog::spdlog
                                            "${TORCH_LIBRARIES}")
target_include_directories(run_self_play
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_compile_options(run_self_play PRIVATE -fopenmp)
target_link_options(run_self_play PRIVATE -fopenmp)

add_executable(benchmark_policy EXCLUDE_FROM_ALL benchmark_policy.cpp)
target_link_libraries(benchmark_policy PRIVATE spdlog::spdlog
                                               "${TORCH_LIBRARIES}")
target_include_directories(benchmark_policy
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
