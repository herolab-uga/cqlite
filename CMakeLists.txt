cmake_minimum_required(VERSION 2.8.3)
project(cqlite))

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  tf
  visualization_msgs
  message_generation
)


add_message_files(
  FILES
  PointArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)


catkin_package( 

CATKIN_DEPENDS message_runtime
)





include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(global_frontier_detector src/global_frontier_detector.cpp src/cqlite_functions.cpp src/cqlite_mtrand.cpp)
target_link_libraries(global_frontier_detector ${catkin_LIBRARIES})

add_executable(local_frontier_detector src/local_frontier_detector.cpp src/cqlte_functions.cpp src/cqlite_mtrand.cpp)
target_link_libraries(local_frontier_detector ${catkin_LIBRARIES})
