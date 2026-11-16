Wanna run this project on Jetson for with the following spec
```
> cat /etc/nv_tegra_release
# R39 (release), REVISION: 2.0, GCID: 45755727, BOARD: generic, EABI: aarch64, DATE: Mon Jun  1 09:28:48 PM UTC 2026
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia

> cat /etc/os-release 
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.4 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
```

I already installed the jetson-containers on the Jetson device and wanna leverage that for the easiest setup. Which jetson-containers commnad to run to build the proper image (including ONNX / CUDA versions, ros2 jazzy packages etc.)