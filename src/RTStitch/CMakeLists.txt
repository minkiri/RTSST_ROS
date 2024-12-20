cmake_minimum_required(VERSION 3.0.2)
project(RTStitch)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

## catkin 설정
catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge
)

## Python 스크립트 설치
catkin_install_python(PROGRAMS
  src/rtstitch_node.py  # ROS 노드 스크립트
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 리소스 파일 설치 (프리트레인된 모델 등)
install(
  DIRECTORY resources/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resources
)

## src 디렉토리도 설치 경로에 포함
install(
  DIRECTORY src/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/src
)

## 의존성 경로 추가
include_directories(
  ${catkin_INCLUDE_DIRS}
)