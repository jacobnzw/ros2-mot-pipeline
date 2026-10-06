cmake_minimum_required(VERSION 3.16)
project(yolo_detector LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime installation prefix")
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS
    "${ONNXRUNTIME_ROOT}/include"
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS
    "${ONNXRUNTIME_ROOT}/lib"
    "${ONNXRUNTIME_ROOT}/lib64"
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime C++ was not found. Set ONNXRUNTIME_ROOT to its installation prefix. "
    "Expected onnxruntime_cxx_api.h and libonnxruntime.so under include/ and lib/."
  )
endif()

add_library(yolo_onnx src/onnx_model.cpp)
target_include_directories(yolo_onnx
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
    "${ONNXRUNTIME_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_onnx PUBLIC "${ONNXRUNTIME_LIBRARY}" ${OpenCV_LIBS})

add_executable(onnx_probe src/onnx_probe.cpp)
target_link_libraries(onnx_probe PRIVATE yolo_onnx)

install(TARGETS yolo_onnx onnx_probe
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)

ament_package()