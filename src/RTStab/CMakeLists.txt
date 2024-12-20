cmake_minimum_required(VERSION 3.0.2)
project(RTStab)

## Find catkin and any required dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  usb_cam
  cv_bridge
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS rospy roscpp std_msgs sensor_msgs image_transport usb_cam cv_bridge
)

## Install Python files
catkin_install_python(PROGRAMS src/rtstab_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Add Python setup
catkin_python_setup()  # 이 줄을 추가하세요

