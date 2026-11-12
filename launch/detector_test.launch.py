from datetime import datetime, timezone
from pathlib import Path

from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch import LaunchContext, LaunchDescription

log_level = LaunchConfiguration("log_level")
log_input_image = LaunchConfiguration("log_input_image")
log_detections = LaunchConfiguration("log_detections")
log_tracks = LaunchConfiguration("log_tracks")
thread_num = LaunchConfiguration("thread_num")
loop = LaunchConfiguration("loop")
seq = LaunchConfiguration("seq")
kitti_root = LaunchConfiguration("kitti_root")
label = LaunchConfiguration("label")
rate = LaunchConfiguration("rate")
precision = LaunchConfiguration("precision")
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
recording_path = PathJoinSubstitution(
    [
        "runs",
        label,
        [TextSubstitution(text=f"{timestamp}_"), seq],
        "recording.rrd",
    ]
)


def generate_launch_description():
    def create_bag_player(context: LaunchContext) -> list[ExecuteProcess | RegisterEventHandler]:
        sequence = seq.perform(context)
        mcap_path = Path(kitti_root.perform(context)).expanduser() / sequence / f"{sequence}.mcap"
        command = ["ros2", "bag", "play", str(mcap_path), "--clock", "--rate", rate.perform(context)]
        if loop.perform(context).lower() == "true":
            command.append("--loop")
        bag_player = ExecuteProcess(cmd=command, name="kitty_player", output="screen")
        shutdown_on_bag_exit = RegisterEventHandler(
            OnProcessExit(
                target_action=bag_player,
                on_exit=[EmitEvent(event=Shutdown(reason="MCAP playback finished"))],
            )
        )
        return [bag_player, shutdown_on_bag_exit]

    def create_tracker_container(context: LaunchContext) -> list[ComposableNodeContainer]:
        model_precision = precision.perform(context).lower()
        model_path = Path(__file__).resolve().parents[1] / "models" / "yolo" / f"yolo11n_{model_precision}.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Detector model does not exist: {model_path}")

        composable_nodes = [
            ComposableNode(
                package="object_tracker",
                plugin="object_tracker::YoloDetectorComponent",
                name="detector",
                parameters=[
                    {
                        "use_sim_time": True,
                        "model_path": str(model_path),
                        "image_topic": "/camera/image_raw/compressed",
                        "detection_topic": "/detections",
                        "log_input_image": log_input_image,
                        "log_detections": log_detections,
                        "recording_path": recording_path,
                    }
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # Timed detections used as observations for forming tracks output as TrackedObjectArray
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
                        "log_tracks": log_tracks,
                        "recording_path": recording_path,
                    }
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ]
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
                "log_detections",
                default_value="false",
                choices=["true", "false"],
                description="Whether to log detector bounding boxes to Rerun.",
            ),
            DeclareLaunchArgument(
                "log_tracks",
                default_value="true",
                choices=["true", "false"],
                description="Whether to log tracker bounding boxes to Rerun.",
            ),
            DeclareLaunchArgument(
                "thread_num",
                default_value="1",
                description="Number of threads to use. If > 1 multi-threaded executor is used.",
            ),
            DeclareLaunchArgument(
                "precision",
                default_value="fp32",
                choices=["fp32", "fp16", "int8", "FP32", "FP16", "INT8"],
                description="Detector model precision; selects models/yolo/yolo11n_<precision>.onnx.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="false",
                choices=["true", "false"],
                description="Whether to loop MCAP playback indefinitely.",
            ),
            DeclareLaunchArgument(
                "rate",
                default_value="1.0",
                description="MCAP playback rate multiplier.",
            ),
            DeclareLaunchArgument(
                "seq",
                default_value="0019",
                description="KITTI sequence number to play.",
            ),
            DeclareLaunchArgument(
                "kitti_root",
                default_value="data/kitti/left_color/training/image_02_mcap",
                description="Root directory containing the sequence MCAP folders.",
            ),
            DeclareLaunchArgument(
                "label",
                default_value="default",
                description="User-provided label included in the Rerun recording directory name.",
            ),
            OpaqueFunction(function=create_bag_player),
            OpaqueFunction(function=create_tracker_container),
        ]
    )
