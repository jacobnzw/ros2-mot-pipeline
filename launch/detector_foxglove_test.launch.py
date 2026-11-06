from datetime import datetime, timezone

from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch import LaunchDescription

recording_path = LaunchConfiguration("recording_path")


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
        executable="component_container",  # multi-threaded: component_container_mt
        composable_node_descriptions=[
            ComposableNode(
                package="object_tracker",
                plugin="object_tracker::YoloDetectorComponent",
                name="detector",
                parameters=[
                    {
                        "use_sim_time": True,
                        "model_path": "models/yolo/yolo11n.onnx",
                        "image_topic": "/camera/image_raw/compressed",
                        "detection_topic": "/detections",
                        "log_input_image": False,
                        "log_detections": False,
                        "recording_path": recording_path,
                    }
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # Converts detections from detection2DArray to ImageMarkerArray that Foxglove supports
            ComposableNode(
                package="object_tracker",
                plugin="object_tracker::YoloDetectorObserver",
                name="detector_observer",
                parameters=[
                    {
                        "use_sim_time": True,
                        "detections_topic": "/detections",
                        "foxglove_topic": "/detection_annotations",
                    }
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    foxglove_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "foxglove_bridge",
            "foxglove_bridge_launch.xml",
            "topic_whitelist:=['/camera/image_raw/compressed', '/detection_annotations']",
        ],
        name="foxglove_bridge",
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "recording_path",
                default_value=(
                    "runs/"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                    "_yolo11n_kitti-0019/recording.rrd"
                ),
                description="Path for the Rerun recording. The parent directory is created automatically.",
            ),
            foxglove_bridge,
            kitty_player,
            tracker_container,
        ]
    )
