project(person_detection)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_BUILD_TYPE Release)

add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Release>:-ffast-math>)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_generation
  roscpp
  sensor_msgs
  std_srvs
  tf
  tf_conversions
  visualization_msgs
)

add_message_files(
  FILES
  PersonDetection.msg
  PersonDetections.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  cv_bridge
  geometry_msgs
  message_runtime
  roscpp
  sensor_msgs
  tf
  tf_conversions
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
