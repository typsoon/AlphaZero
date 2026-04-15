cmake_minimum_required(VERSION 3.10)
project(MyTorchProject LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address,undefined")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
# set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address,undefined")

# Find LibTorch
find_package(Torch REQUIRED)

# Workaround for missing nvToolsExt
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
  set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "")
endif()

# Use link paths automatically for RPATH so that it works dynamically based on Torch_DIR
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add the executable

project(my_ai_engine)
add_subdirectory(engine)
add_subdirectory(training)
add_subdirectory(inference_server)

# add_executable(self_play training/self_play.cpp)
# target_link_libraries(self_play
#   PRIVATE engine
#   "${TORCH_LIBRARIES}"
# )


# Enable CUDA if available (optional, just for clarity)
if(TORCH_CUDA_FOUND)
  message(STATUS "Compiling with CUDA support")
else()
  message(STATUS "Compiling without CUDA support")
endif()

