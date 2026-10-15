if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file ]]; then
  print -u2 "Source this file from zsh: source scripts/ros2_env.zsh"
  return 1
fi

_ros2_env_script="${(%):-%N}"
_ros2_env_root="${_ros2_env_script:A:h:h}"

if [[ ! -f /opt/ros/jazzy/setup.zsh ]]; then
  print -u2 "ROS 2 Jazzy setup not found at /opt/ros/jazzy/setup.zsh"
  return 1
fi
source /opt/ros/jazzy/setup.zsh || return 1

if [[ -f "$_ros2_env_root/install/setup.zsh" ]]; then
  source "$_ros2_env_root/install/setup.zsh" || return 1
fi

export ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-$HOME/opt/onnxruntime-linux-x64-gpu_cuda12-1.30.0}"
export CUDNN_ROOT="${CUDNN_ROOT:-$HOME/opt/cuda12-cudnn-runtime}"
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export CUDNN_LIBRARY_DIR="${CUDNN_LIBRARY_DIR:-$CUDNN_ROOT/nvidia/cudnn/lib}"

if [[ ! -f "$ONNXRUNTIME_ROOT/lib/libonnxruntime.so" ||
      ! -f "$ONNXRUNTIME_ROOT/lib/libonnxruntime_providers_cuda.so" ]]; then
  print -u2 "ONNX Runtime CUDA libraries not found under $ONNXRUNTIME_ROOT"
  return 1
fi
if [[ ! -e "$CUDNN_LIBRARY_DIR/libcudnn.so" ]]; then
  print -u2 "libcudnn.so not found under $CUDNN_LIBRARY_DIR"
  return 1
fi

_ros2_prepend_library_path() {
  [[ -d "$1" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$1:"*) ;;
    *) export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
}

_ros2_prepend_library_path "$ONNXRUNTIME_ROOT/lib"
_ros2_prepend_library_path "$CUDA_ROOT/lib64"
_ros2_prepend_library_path "$CUDNN_LIBRARY_DIR"

if [[ -d /usr/lib/wsl/drivers ]]; then
  _ros2_wsl_driver_file="$(find /usr/lib/wsl/drivers -name libnvidia-ptxjitcompiler.so.1 -print -quit 2>/dev/null)"
  if [[ -z "$_ros2_wsl_driver_file" ]]; then
    print -u2 "Could not locate libnvidia-ptxjitcompiler.so.1 under /usr/lib/wsl/drivers"
    unfunction _ros2_prepend_library_path
    return 1
  fi
  export WSL_NVIDIA_DRIVER_ROOT="${_ros2_wsl_driver_file:h}"
  _ros2_prepend_library_path "$WSL_NVIDIA_DRIVER_ROOT"
fi

unfunction _ros2_prepend_library_path
unset _ros2_env_script _ros2_env_root _ros2_wsl_driver_file
