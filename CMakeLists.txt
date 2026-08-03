cmake_minimum_required(VERSION 3.0.2)
project(limo_person_following)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

add_message_files(
  FILES
  PersonDetection.msg
  PersonDetections.msg
  TargetPerson.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)
