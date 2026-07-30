find_package(Torch REQUIRED)
file(GLOB INF_SRC CONFIGURE_DEPENDS *.cpp)
add_library(inference STATIC ${INF_SRC})

if(NOT Python3_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# Attempt to find libtorchtrt.so using Python
set(_trt_lib_discovery_script
    "import os, torch_tensorrt\n\
print('TRT_PATH=' +\n\
      os.path.join(os.path.dirname(torch_tensorrt.__file__), 'lib',\n\
                   'libtorchtrt.so'))")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "${_trt_lib_discovery_script}"
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
  set(_trt_runtime_libs_discovery_script
      "import os, nvidia, torch_tensorrt\n\
trt = os.path.dirname(os.path.dirname(torch_tensorrt.__file__))\n\
print('TRT_LIBS_PATH=' + os.path.join(trt, 'tensorrt_libs') + ';' +\n\
      os.path.join(nvidia.__path__[0], 'cuda_runtime', 'lib'))")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "${_trt_runtime_libs_discovery_script}"
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

# Native raw-TensorRT-engine loading (network.py's backend="onnx" checkpoints -
# see basic_infer.cpp's TensorRTInferenceBackend). Needs the TensorRT C++ SDK
# headers (NvInfer.h), which pip's `tensorrt` package doesn't ship (Python
# bindings + runtime .so only) - set
# -DALPHAZERO_TENSORRT_INCLUDE_DIR=/path/to/TensorRT-X.Y.Z/include (a manually
# downloaded SDK - see the machine-local skill) to enable. Only actually
# turned on when the header version's major matches the active venv's pip
# `tensorrt` runtime major version too: TensorRT's C++ ABI isn't guaranteed
# stable across major versions, so compiling classes against one major
# version's headers and then linking/running against a different major
# version's runtime .so isn't safe.
set(ALPHAZERO_TENSORRT_INCLUDE_DIR
    ""
    CACHE PATH "Path to a TensorRT SDK's include/ dir (NvInfer.h)")

if(ALPHAZERO_TENSORRT_INCLUDE_DIR AND EXISTS
                                       "${ALPHAZERO_TENSORRT_INCLUDE_DIR}/NvInfer.h")
  set(_trt_pylib_script
      "import tensorrt as trt, tensorrt_libs, os\n\
print('TRT_RUNTIME_MAJOR=' + trt.__version__.split('.')[0])\n\
print('TRT_RUNTIME_LIBDIR=' + os.path.dirname(tensorrt_libs.__file__))")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "${_trt_pylib_script}"
    RESULT_VARIABLE _trt_py_found
    OUTPUT_VARIABLE _trt_py_out
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  string(REGEX MATCH "TRT_RUNTIME_MAJOR=([0-9]+)" _m1 "${_trt_py_out}")
  set(_trt_runtime_major "${CMAKE_MATCH_1}")
  string(REGEX MATCH "TRT_RUNTIME_LIBDIR=([^\n]*)" _m2 "${_trt_py_out}")
  set(_trt_runtime_libdir "${CMAKE_MATCH_1}")

  file(STRINGS "${ALPHAZERO_TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _hdr_major_lines
       REGEX "define TRT_MAJOR_ENTERPRISE|define TRT_MAJOR ")
  string(REGEX MATCH "([0-9]+)" _hdr_major "${_hdr_major_lines}")

  if(_trt_py_found EQUAL 0
     AND _trt_runtime_libdir
     AND _hdr_major STREQUAL "${_trt_runtime_major}")
    message(
      STATUS "Native TensorRT engine loading ENABLED: headers+runtime both major v${_hdr_major}"
    )
    # PUBLIC (not PRIVATE): consumers that also want to touch the TensorRT
    # C++ API directly - e.g. engine/tests/test_inference_backend.cpp's
    # native-engine test fixture - need this include dir too, not just the
    # ALPHAZERO_NATIVE_TRT_ENGINE compile definition below.
    target_include_directories(inference PUBLIC "${ALPHAZERO_TENSORRT_INCLUDE_DIR}")
    target_link_libraries(inference "${_trt_runtime_libdir}/libnvinfer.so.${_hdr_major}")
    target_compile_definitions(inference PUBLIC ALPHAZERO_NATIVE_TRT_ENGINE=1)
    set_property(
      TARGET inference
      APPEND
      PROPERTY BUILD_RPATH "${_trt_runtime_libdir}")
    set_property(
      TARGET inference
      APPEND
      PROPERTY INSTALL_RPATH "${_trt_runtime_libdir}")
  else()
    message(
      STATUS
        "Native TensorRT engine loading disabled: header major=${_hdr_major} vs "
        "runtime major=${_trt_runtime_major} (need pip 'tensorrt' installed with a "
        "matching major version)")
  endif()
else()
  message(STATUS "Native TensorRT engine loading disabled: ALPHAZERO_TENSORRT_INCLUDE_DIR not set")
endif()

target_include_directories(inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
