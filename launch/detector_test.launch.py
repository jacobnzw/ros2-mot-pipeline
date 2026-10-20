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
                    }
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # Detections from detection2DArray used as observations for forming tracks output as TrackedObjectArray
            ComposableNode(
                package="object_tracker",
                plugin="object_tracker::TrackerComponent",
                name="tracker",
                parameters=[
                    {
                        "use_sim_time": True,
                        "detections_topic": "/detections",
                        "tracks_topic": "~/tracks",
                        "high_threshold": 0.25,  # detections above this take part in the first association
                        "low_threshold": 0.1,  # (low, high) detections only rescue existing tracks
                        "new_track_threshold": 0.25,  # minimum score to start a new track
                        "match_threshold": 0.8,  # max (1 - IoU) cost in the first association
                        "track_buffer": 10,  # frames a lost track is kept for re-identification
                        "frame_rate": 10.0,
                        "fuse_score": True,
                    }
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    return LaunchDescription([kitty_player, tracker_container])
