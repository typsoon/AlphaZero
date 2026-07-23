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

  add_executable(test_replay_buffer tests/test_replay_buffer.cpp)
  target_link_libraries(test_replay_buffer PRIVATE engine CppUTest
                                                   "${TORCH_LIBRARIES}")
  target_compile_definitions(test_replay_buffer
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)

  add_executable(test_inference_cache tests/test_inference_cache.cpp)
  target_link_libraries(test_inference_cache PRIVATE engine CppUTest
                                                     "${TORCH_LIBRARIES}")
  target_compile_definitions(test_inference_cache
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)

  add_executable(test_resignation tests/test_resignation.cpp)
  target_link_libraries(test_resignation PRIVATE engine CppUTest)
  target_compile_definitions(test_resignation
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)

  add_executable(test_state_encoder tests/test_state_encoder.cpp)
  target_link_libraries(test_state_encoder PRIVATE engine CppUTest
                                                   "${TORCH_LIBRARIES}")
  target_compile_definitions(test_state_encoder
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)

  add_executable(test_chess_encoder_v2history
                 tests/test_chess_encoder_v2history.cpp)
  target_link_libraries(test_chess_encoder_v2history
                        PRIVATE engine CppUTest "${TORCH_LIBRARIES}")
  target_compile_definitions(test_chess_encoder_v2history
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)
endif()

# ---------------------------------------------------------------------------
# Standalone AddressSanitizer regression test for the Chess pawn-diagonal-
# capture out-of-bounds board read (see the i==0/i==7 guards' comments in
# move_rules_P()/move_rules_p(), engine/game/chess.cpp) - a heap-buffer-overflow
# ASan reliably catches, unlike the DynamicBatcher race below which needs TSan.
# Built as its own target from its own explicit source list, same reasoning as
# ALPHAZERO_BUILD_TSAN_TEST below: ASan is incompatible with the main build's
# flags (and with TSan), so this is off by default rather than a global change.
option(ALPHAZERO_BUILD_ASAN_CHESS_TEST
       "Build the standalone AddressSanitizer Chess pawn-OOB regression test"
       OFF)
if(ALPHAZERO_BUILD_ASAN_CHESS_TEST)
  find_package(CppUTest CONFIG REQUIRED)
  add_executable(test_chess_asan tests/test_chess.cpp game/chess.cpp
                                 game/chess_encoder.cpp game/bitboard.cpp)
  target_include_directories(
    test_chess_asan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/game
                            ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
  target_link_libraries(test_chess_asan PRIVATE "${TORCH_LIBRARIES}" CppUTest
                                                spdlog::spdlog)
  target_compile_definitions(test_chess_asan
                             PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=1)
  target_compile_options(test_chess_asan PRIVATE -fsanitize=address,undefined
                                                 -g -fno-omit-frame-pointer)
  target_link_options(test_chess_asan PRIVATE -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Standalone AddressSanitizer/UBSan self-play stress harness for the
# ChessEncoderV2History path (see tests/test_selfplay_stress_asan.cpp). Off by
# default and built as its own target from an explicit source list - same
# reasoning as the ASan chess test and the TSan test below: ASan is incompatible
# with the main build's flags. Compiles the encoder, Chess history hook, and the
# DynamicBatcher pipeline directly so ASan instruments every write into the
# encoder's output buffers.
option(
  ALPHAZERO_BUILD_ASAN_STRESS_TEST
  "Build the standalone AddressSanitizer ChessEncoderV2History self-play stress test"
  OFF)
if(ALPHAZERO_BUILD_ASAN_STRESS_TEST)
  add_executable(
    test_selfplay_stress_asan
    tests/test_selfplay_stress_asan.cpp
    inference/basic_infer.cpp
    inference/inference_cache.cpp
    game/chess.cpp
    game/chess_encoder.cpp
    game/chess_encoder_v2history.cpp
    game/connect4.cpp
    game/connect4_encoder.cpp
    game/encoder_factory.cpp
    game/bitboard.cpp)
  target_include_directories(
    test_selfplay_stress_asan
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inference
            ${CMAKE_CURRENT_SOURCE_DIR}/game ${CMAKE_CURRENT_SOURCE_DIR}
            ${TORCH_INCLUDE_DIRS})
  target_link_libraries(test_selfplay_stress_asan PRIVATE "${TORCH_LIBRARIES}"
                                                          spdlog::spdlog)
  # -fno-sanitize=vptr: UBSan's vptr check emits references to typeinfo for
  # libtorch's JIT types (e.g. c10::DynamicType) that the pip-distributed
  # libtorch does not export, breaking the link. The memory-corruption hunt this
  # binary exists for is ASan's job; the rest of UBSan is kept.
  target_compile_options(
    test_selfplay_stress_asan
    PRIVATE -fsanitize=address,undefined -fno-sanitize=vptr -g
            -fno-omit-frame-pointer)
  target_link_options(test_selfplay_stress_asan PRIVATE
                      -fsanitize=address,undefined -fno-sanitize=vptr)
endif()

# ---------------------------------------------------------------------------
# Standalone ThreadSanitizer regression test for the DynamicBatcher buffer-slot
# race (see worker_loop()'s comment in inference/basic_infer.cpp and
# tests/test_dynamic_batcher_tsan.cpp). Off by default and built as its own
# target from its own explicit source list - NOT linked against the regular
# `game`/`inference` static libraries, which aren't compiled with
# -fsanitize=thread and so wouldn't let TSan see accesses (e.g.
# GameState::write_canonical_state()'s write into the shared pinned buffer) that
# happen inside them. TSan is its own build mode (incompatible with ASan/UBSan
# and with the main build's flags), hence a dedicated option rather than a
# global compiler-flag change.
option(
  ALPHAZERO_BUILD_TSAN_TEST
  "Build the standalone ThreadSanitizer DynamicBatcher race regression test"
  OFF)
if(ALPHAZERO_BUILD_TSAN_TEST)
  add_executable(
    test_dynamic_batcher_tsan
    tests/test_dynamic_batcher_tsan.cpp
    inference/basic_infer.cpp
    inference/inference_cache.cpp
    game/connect4.cpp
    game/connect4_encoder.cpp
    game/chess.cpp
    game/chess_encoder.cpp
    game/encoder_factory.cpp
    game/bitboard.cpp)
  target_include_directories(
    test_dynamic_batcher_tsan
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inference
            ${CMAKE_CURRENT_SOURCE_DIR}/game ${CMAKE_CURRENT_SOURCE_DIR}
            ${TORCH_INCLUDE_DIRS})
  target_link_libraries(test_dynamic_batcher_tsan PRIVATE "${TORCH_LIBRARIES}"
                                                          spdlog::spdlog)
  target_compile_options(test_dynamic_batcher_tsan
                         PRIVATE -fsanitize=thread -g -fno-omit-frame-pointer)
  target_link_options(test_dynamic_batcher_tsan PRIVATE -fsanitize=thread)
endif()
