project(person_detection)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_BUILD_TYPE Release)

add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Release>:-ffast-math>)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  moveit_core
  moveit_ros_planning
  moveit_ros_move_group
  tf
  tf_conversions
  roscpp
  std_srvs
  moveit_ros_planning_interface
  visualization_msgs
  interactive_markers
  moveit_ros_robot_interaction
  sensor_msgs
  cv_bridge
  apriltags
)

catkin_package(
  CATKIN_DEPENDS
  moveit_core
  moveit_ros_planning
  roscpp
  tf
  tf_conversions
  sensor_msgs
  cv_bridge
  apriltags
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(Boost REQUIRED COMPONENTS program_options)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

#add_executable(apriltag_command_cards src/apriltag_command_cards.cpp)
#add_dependencies(apriltag_command_cards ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_link_libraries(apriltag_command_cards yaml-cpp ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

