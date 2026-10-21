cmake_minimum_required(VERSION 3.16)
project(object_tracker LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(foxglove_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc dnn)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

# Fetch ReRun C++ SDK
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(rerun_sdk)
set_target_properties(rerun_sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pipeline_metrics SHARED src/metrics.cpp)
target_include_directories(pipeline_metrics
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(pipeline_metrics PRIVATE rerun_sdk fmt::fmt)
ament_target_dependencies(pipeline_metrics PUBLIC rclcpp sensor_msgs vision_msgs)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime installation prefix")

# Flag to enable CUDA build, default ON
option(YOLO_ENABLE_CUDA "Use the ONNX Runtime CUDA execution provider" ON)
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

if(YOLO_ENABLE_CUDA)
  find_library(ONNXRUNTIME_CUDA_PROVIDER_LIBRARY
    NAMES onnxruntime_providers_cuda
    HINTS
    "${ONNXRUNTIME_ROOT}/lib"
    "${ONNXRUNTIME_ROOT}/lib64"
  )
  if(NOT ONNXRUNTIME_CUDA_PROVIDER_LIBRARY)
    message(FATAL_ERROR
      "YOLO_ENABLE_CUDA is ON, but libonnxruntime_providers_cuda was not found under "
      "ONNXRUNTIME_ROOT. Install the ONNX Runtime GPU distribution and set ONNXRUNTIME_ROOT to it."
    )
  endif()
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime C++ was not found. Set ONNXRUNTIME_ROOT to its installation prefix. "
    "Expected onnxruntime_cxx_api.h and libonnxruntime.so under include/ and lib/."
  )
endif()


# ---------- Custom messages ----------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedObject.msg"
  "msg/TrackedObjectArray.msg"
  DEPENDENCIES std_msgs
)
# Make the generated types visible to the rest of the package
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ---------- YOLO Detector ----------
add_library(yolo_onnx src/detector/onnx_model.cpp)
set_target_properties(yolo_onnx PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(YOLO_ENABLE_CUDA)
  target_compile_definitions(yolo_onnx PRIVATE YOLO_ENABLE_CUDA=1)
  message(STATUS "ONNX Runtime CUDA execution provider enabled")
else()
  message(STATUS "ONNX Runtime CUDA execution provider disabled (CPU execution)")
endif()
target_include_directories(yolo_onnx
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  "${ONNXRUNTIME_INCLUDE_DIR}"
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_onnx PUBLIC "${ONNXRUNTIME_LIBRARY}" ${OpenCV_LIBS})

add_executable(onnx_probe src/detector/onnx_probe.cpp)
target_link_libraries(onnx_probe PRIVATE yolo_onnx fmt::fmt)

add_library(yolo_detector_component SHARED src/detector/yolo_detector_component.cpp)
target_include_directories(yolo_detector_component
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  "${ONNXRUNTIME_INCLUDE_DIR}"
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_detector_component PUBLIC yolo_onnx PRIVATE pipeline_metrics)
ament_target_dependencies(yolo_detector_component
  PUBLIC
  rclcpp
  rclcpp_components
  sensor_msgs
  vision_msgs
  cv_bridge
)
rclcpp_components_register_nodes(yolo_detector_component "object_tracker::YoloDetectorComponent")

# ---------- Observer for YOLO Detector  ----------
add_library(yolo_detector_observer SHARED src/detector/yolo_detector_observer.cpp)
target_include_directories(yolo_detector_observer
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(yolo_detector_observer)
ament_target_dependencies(yolo_detector_observer
  PUBLIC
  rclcpp
  rclcpp_components
  vision_msgs
  foxglove_msgs
)
rclcpp_components_register_nodes(yolo_detector_observer "object_tracker::YoloDetectorObserver")

# ---------- ByteTrack Tracker ----------
add_library(tracker_component SHARED
  src/tracker/byte_tracker.cpp
  src/tracker/kalman_filter.cpp
  src/tracker/linear_assignment.cpp
  src/tracker/tracker_component.cpp
)
target_include_directories(tracker_component
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(tracker_component PRIVATE "${cpp_typesupport_target}" Eigen3::Eigen pipeline_metrics)
ament_target_dependencies(tracker_component
  PUBLIC
  rclcpp
  rclcpp_components
  std_msgs
  vision_msgs
)
rclcpp_components_register_nodes(tracker_component "object_tracker::TrackerComponent")

install(TARGETS yolo_onnx onnx_probe pipeline_metrics yolo_detector_component yolo_detector_observer tracker_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)

ament_export_dependencies(Eigen3)
ament_package()
