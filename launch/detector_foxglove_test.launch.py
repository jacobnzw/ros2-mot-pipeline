from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch import LaunchDescription


def generate_launch_description():

    mcap_path = "data/kitti/left_color/mcap/0019/0019.mcap"
    kitty_player = ExecuteProcess(
        cmd=["ros2", "bag", "play", mcap_path, "--clock", "--rate", "1.0"],
        name="kitty_player",
        output="screen",
    )

    tracker_container = ComposableNodeContainer(
        name="tracker_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="yolo_detector",
                plugin="yolo_detector::YoloDetectorComponent",
                name="detector",
                parameters=[
                    {
                        "model_path": "models/yolo/yolo11n.onnx",
                        "image_topic": "/camera/image_raw/compressed",
                        "detection_topic": "/detections",
                    }
                ],
            ),
            # Converts detections from detection2DArray to ImageMarkerArray that Foxglove supports
            ComposableNode(
                package="yolo_detector",
                plugin="yolo_detector::YoloDetectorObserver",
                name="detector_observer",
                parameters=[
                    {
                        "detections_topic": "/detections",
                        "foxglove_topic": "/detection_annotations",
                    }
                ],
            ),
        ],
    )

    foxglove_bridge = ExecuteProcess(
        cmd=["ros2", "launch", "foxglove_bridge", "foxglove_bridge_launch.xml"],
        name="foxglove_bridge",
        output="screen",
    )

    return LaunchDescription([foxglove_bridge, kitty_player, tracker_container])
