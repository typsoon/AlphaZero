find_package(Torch REQUIRED)
file(GLOB INF_SRC CONFIGURE_DEPENDS *.cpp)
add_library(inference STATIC ${INF_SRC})

if(NOT Python3_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# Attempt to find libtorchtrt.so using Python
set(_TRT_LIB_DISCOVERY_SCRIPT
    "import os, torch_tensorrt; print('TRT_PATH=' + os.path.join(os.path.dirname(torch_tensorrt.__file__), 'lib', 'libtorchtrt.so'))"
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "${_TRT_LIB_DISCOVERY_SCRIPT}"
  RESULT_VARIABLE TRT_FOUND
  OUTPUT_VARIABLE TRT_LIB_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

string(REGEX MATCH "TRT_PATH=([^\n]*)" _matched_trt "${TRT_LIB_OUTPUT}")
set(TRT_LIB_PATH ${CMAKE_MATCH_1})

if(TRT_FOUND EQUAL 0 AND EXISTS "${TRT_LIB_PATH}")
  message(STATUS "Found torch_tensorrt: ${TRT_LIB_PATH}")
  target_link_libraries(inference "${TORCH_LIBRARIES}" "${TRT_LIB_PATH}" game
                        spdlog::spdlog)
  target_link_libraries(inference "${CMAKE_DL_LIBS}")
  target_compile_definitions(inference
                             PUBLIC ALPHAZERO_TRT_LIB_PATH="${TRT_LIB_PATH}")

  # Add tensorrt_libs and torch libs to RPATH
  set(_TRT_RUNTIME_LIBS_DISCOVERY_SCRIPT
      "import os, nvidia, torch_tensorrt\n\
trt = os.path.dirname(os.path.dirname(torch_tensorrt.__file__))\n\
print('TRT_LIBS_PATH=' + os.path.join(trt, 'tensorrt_libs') + ';' +\n\
      os.path.join(nvidia.__path__[0], 'cuda_runtime', 'lib'))")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "${_TRT_RUNTIME_LIBS_DISCOVERY_SCRIPT}"
    RESULT_VARIABLE TRT_LIBS_FOUND
    OUTPUT_VARIABLE TRT_LIBS_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  string(REGEX MATCH "TRT_LIBS_PATH=([^\n]*)" _matched_libs
               "${TRT_LIBS_OUTPUT}")
  set(TRT_LIBS_PATH ${CMAKE_MATCH_1})

  if(TRT_LIBS_FOUND EQUAL 0 AND NOT "${TRT_LIBS_PATH}" STREQUAL "")
    message(STATUS "Adding TensorRT backend to RPATH: ${TRT_LIBS_PATH}")
    set_property(
      TARGET inference
      APPEND
      PROPERTY BUILD_RPATH "${TRT_LIBS_PATH}")
    set_property(
      TARGET inference
      APPEND
      PROPERTY INSTALL_RPATH "${TRT_LIBS_PATH}")
  endif()
else()
  message(STATUS "torch_tensorrt not found, falling back to standard Torch")
  target_link_libraries(inference "${TORCH_LIBRARIES}" game spdlog::spdlog)
endif()

target_include_directories(inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
