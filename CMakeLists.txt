cmake_minimum_required(VERSION 3.10)
project(MyTorchProject LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address,undefined")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g") set(CMAKE_LINKER_FLAGS
# "${CMAKE_LINKER_FLAGS} -fsanitize=address,undefined")

# Find LibTorch
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Workaround for missing nvToolsExt
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
  set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "")
endif()

# Use link paths automatically for RPATH so that it works dynamically based on
# Torch_DIR
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Detect TensorRT + CUDA runtime library paths and bake them into RPATH of all
# executables so they can load TRT engines without needing LD_LIBRARY_PATH.
set(_TRT_RPATH_DISCOVERY_SCRIPT
    "import os; import nvidia, torch_tensorrt; trt=os.path.dirname(os.path.dirname(torch_tensorrt.__file__)); print(os.path.join(trt,'tensorrt_libs') + ';' + os.path.join(nvidia.__path__[0],'cuda_runtime','lib'))"
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "${_TRT_RPATH_DISCOVERY_SCRIPT}"
  RESULT_VARIABLE _trt_rpath_found
  OUTPUT_VARIABLE _trt_rpath_raw
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX MATCH "[^\n]*$" _trt_rpath "${_trt_rpath_raw}")
if(_trt_rpath_found EQUAL 0 AND NOT "${_trt_rpath}" STREQUAL "")
  message(STATUS "TensorRT runtime RPATH: ${_trt_rpath}")
  list(APPEND CMAKE_BUILD_RPATH ${_trt_rpath})
  list(APPEND CMAKE_INSTALL_RPATH ${_trt_rpath})
else()
  message(STATUS "TensorRT not found, skipping TRT RPATH")
endif()

# Use vcpkg to find spdlog
find_package(spdlog REQUIRED)

# Add the executable

project(my_ai_engine)
add_subdirectory(engine)
add_subdirectory(training)
add_subdirectory(inference_server)

# add_executable(self_play training/self_play.cpp)
# target_link_libraries(self_play PRIVATE engine "${TORCH_LIBRARIES}" )

# Enable CUDA if available (optional, just for clarity)
if(TORCH_CUDA_FOUND)
  message(STATUS "Compiling with CUDA support")
else()
  message(STATUS "Compiling without CUDA support")
endif()
