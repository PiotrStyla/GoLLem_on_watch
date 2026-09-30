cmake_minimum_required(VERSION 3.22)
project(watchllm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# upstream llama.cpp, cloned+patched by tools/setup_llamacpp.*
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
# NDK already targets the right ARM features per ABI; host tuning would break
# cross-compilation (same class of bug as the watchOS scalar-kernel fallback).
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../vendor/llama.cpp llama.cpp)

add_library(watchllm SHARED llm_jni.cpp)
target_link_libraries(watchllm PRIVATE llama android log)
