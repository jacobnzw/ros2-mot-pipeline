from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch import LaunchDescription, LaunchContext

log_level = LaunchConfiguration("log_level")
log_input_image = LaunchConfiguration("log_input_image")
thread_num = LaunchConfiguration("thread_num")


def generate_launch_description():

    mcap_path = "data/kitti/left_color/mcap/0019/0019.mcap"
    kitty_player = ExecuteProcess(
        cmd=["ros2", "bag", "play", mcap_path, "--clock", "--rate", "1.0"],
        name="kitty_player",
        output="screen",
    )

    composable_nodes = [
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
                    "log_input_image": log_input_image,
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
    ]

    def create_tracker_container(context: LaunchContext) -> list[ComposableNodeContainer]:
        thread_count = int(thread_num.perform(context))
        arguments = ["--ros-args", "--log-level", log_level.perform(context)]
        executable = "component_container"
        if thread_count > 1:
            executable = "component_container_mt"
            arguments.extend(["-p", f"thread_num:={thread_count}"])

        return [
            ComposableNodeContainer(
                name="tracker_container",
                namespace="",
                package="rclcpp_components",
                executable=executable,
                arguments=arguments,
                # NOTE: Useful for experimentation. Although, core pinning makes things worse from experience.
                # prefix="taskset -c 2-5",  # Pin the container process to cores 2-5.
                # prefix="chrt --fifo 10",  # Change the container process scheduling policy to FIFO
                composable_node_descriptions=composable_nodes,
            )
        ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                choices=["debug", "info", "warn", "error", "fatal"],
                description="Log level.",
            ),
            DeclareLaunchArgument(
                "log_input_image",
                default_value="false",
                choices=["true", "false"],
                description="Whether to log the detector input image (set to true to enable).",
            ),
            DeclareLaunchArgument(
                "thread_num",
                default_value="1",
                description="Number of threads to use. If > 1 multi-threaded executor is used.",
            ),
            kitty_player,
            OpaqueFunction(function=create_tracker_container),
        ]
    )
