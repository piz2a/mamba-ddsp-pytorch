cmake_minimum_required(VERSION 3.18)
project(ddsp_onnx_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an extracted ONNX Runtime C++ release.")
endif()

add_executable(ddsp_onnx_benchmark benchmark.cpp)
target_include_directories(ddsp_onnx_benchmark PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_link_directories(ddsp_onnx_benchmark PRIVATE "${ONNXRUNTIME_ROOT}/lib")
target_link_libraries(ddsp_onnx_benchmark PRIVATE onnxruntime)
set_target_properties(ddsp_onnx_benchmark PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
