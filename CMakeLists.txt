project(person_detection)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_BUILD_TYPE Release)

add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Release>:-ffast-math>)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  tf
  tf_conversions
  roscpp
  std_srvs
  visualization_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

add_message_files(FILES PersonDetection.msg PersonDetections.msg)

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  tf
  tf_conversions
  sensor_msgs
  cv_bridge
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
