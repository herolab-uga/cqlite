# CQLite
It is a ROS package that implements a CQLite: Coverage-biased Q-Learning Lite for Efficient Multi-Robot Exploration algorithm for mobile robots. It uses occupancy grid as a map representation. 


The following figure shows the oevrview of CQLite exploration:

![alt text](/image/cqlite_overview.png "System Overview of CQLite")

### About package repository
The packgae has 5 different ROS nodes:

  - Global frontier point detector node.
  
  - Local frontier point detector node.
  
  - CQLite filter node.
  
  - CQLite planner node.
  
  - opencv-based frontier detector node.

## 1. Requirements
The package has been tested on both ROS Noetic and ROS Melodic for both simulated and hardware Turtlebot3 robot (burger). The following requirements are needed before installing the package:

1- You should have installed a ROS distribution (indigo or later. Recommended is either noetic or melodic).

2- Created a workspace.

3- Install the "gmapping" ROS package: on PC and each robot, as:

```sh
$ sudo apt-get install ros-noetic-gmapping
```
4- Install the ROS navigation stack. You can do that with the following command (assuming Ubuntu, ROS Kinetic):
```sh
$ sudo apt-get install ros-noetic-navigation
```
5- You should have Python 3.6+

6-You should have/install the following python modules:

-OpenCV (cv2)
```sh
$ sudo apt-get install python-opencv
```
-Numpy
```sh
$ sudo apt-get install python-numpy
```
-Sklearn
```sh
$ sudo apt-get install python-scikits-learn
```
## 2. Installation
Download the package and place it inside the ```/src``` folder of your ROS workspace. And then compile using ```catkin_make```.

## 3. Setting Up Your Robots
This package provides an exploration strategy for cooperative robot. However, for it to work, you should have set your robots ready using the [navigation stack](http://wiki.ros.org/navigation). Additionally, the robots must be set and prepared as follows.

### 3.1. Robots Network
For the cooperative robotic configuration, the package doesn't require special network configuration, it simply works by having a single ROS master (can be one of the robots). So on the other robots, the ```ROS_MASTER_URI``` parameter should be pointing at the master's address. 
For more information on setting up ROS on multiple machines, follow [this](http://wiki.ros.org/ROS/NetworkSetup) link.

### 3.2. Robot's frame names in ```tf```
All robot's frames should be prefixed by its name. Naming of robots starts from "/tb3_0", "/tb3_1", ... and so on.

### 3.3. Robot's node and topic names
All the nodes and topics running on a robot must also be prefixed by its name. For tb3_0, node names should look like:  ```/tb3_0/slam_gmapping```.

And topic names should be like: ```/tb3_0/odom```,  ```/tb3_0/map```,  ```/tb3_0/scan```, ..etc.


### 3.5. A mapping node
Each robot should have a local map generated from the [gmapping](http://wiki.ros.org/gmapping) package.

### 3.6. A map merging node
For the multi-robot case, there should be a node that merges all the local maps into one global map. You can use [this](http://wiki.ros.org/multirobot_map_merge) package.

## 4. Nodes
There are 4 types of nodes; nodes for detecting frontier points in an occupancy grid map, a node for filtering the detected points, a node for assigning the points to the robots, and a node for planning path for robot. The following figure shows the system arcitecture of CQLite:

![alt text](/image/cqlite_architecture.png "System Achitecture of CQLite")

### 4.1. global_frontier_detector
The ```global_frontier_detector``` node takes an occupancy grid and finds frontier points (which are exploration targets) in it. It publishes the detected points so the filter node can process. 

#### 4.1.1. Parameters
 - ```~map_topic``` (string, default: "/tb3_0/map"): This parameter defines the topic name on which the node will recieve the map.

#### 4.1.2. Subscribed Topics
 - The map (Topic name is defined by the ```~map_topic``` parameter) ([nav_msgs/OccupancyGrid](http://docs.ros.org/api/nav_msgs/html/msg/OccupancyGrid.html))

- ```clicked_point``` ([geometry_msgs/PointStamped Message](http://docs.ros.org/api/geometry_msgs/html/msg/PointStamped.html)): The ```global_frontier_detector``` node requires that the region to be explored is defined. This topic is where the node recieves five points that define the region. The first four points are four defining a square region to be explored, and the last point is the tree starting point. After publishing those five points on this topic, the CQLite will start detecting frontier points. The five points are intended to be published from Rviz using ![alt text](https://github.com/hasauino/storage/blob/master/pictures/publishPointRviz_button.png "Rviz publish point button") button.

#### 4.1.3. Published Topics
 - ```detected_points``` ([geometry_msgs/PointStamped Message](http://docs.ros.org/api/geometry_msgs/html/msg/PointStamped.html)): The topic on which the node publishes detected frontier points.

- ```~shapes``` ([visualization_msgs/Marker Message](http://docs.ros.org/api/visualization_msgs/html/msg/Marker.html)): On this topic, the node publishes line shapes to visualize the CQLite using Rviz.


### 4.2. local_frontier_detector
This node is similar to the global_frontier_detector. However, it works differently, as a connected graph here keeps resetting every time a frontier point is detected. This node is intended to be run along side the global__frontier_detector node, it is responsible for fast detection of frontier points that lie in the close vicinity of the robot.

All detectors will be publishing detected frontier points on the same topic (```/detected_points```).
#### 4.2.1. Parameters
- ```~robot_frame``` (string, default: "/tb3_0/base_link"): The frame attached to the robot. Every time the tree resets, it will start from the current robot location obtained from this frame.

 - ```~map_topic``` (string, default: "/tb3_0/map"): This parameter defines the topic name on which the node will recieve the map.

#### 4.2.2. Subscribed Topics
 - The map (Topic name is defined by the ```~map_topic``` parameter) ([nav_msgs/OccupancyGrid](http://docs.ros.org/api/nav_msgs/html/msg/OccupancyGrid.html)).

- ```clicked_point``` ([geometry_msgs/PointStamped Message](http://docs.ros.org/api/geometry_msgs/html/msg/PointStamped.html)): The ```lobal_frontier_detector``` also subscribes to this topic similar to the global_frontier_detector. 
#### 4.2.3. Published Topics
 - Same as global frontier detector with local prefix


### 4.3. frontier_opencv_detector
This node is another frontier detector, but it is not based on CQLite. This node uses OpenCV tools to detect frontier points. It is intended to be run alone, and only one instance should be run.

Originally this node was implemented for comparison against the standard frontier detectors. Running this node along side the CQLite detectors (local and global) may enhance the speed of frontier points detection.

#### 4.3.1. Parameters
 - ```~map_topic``` (string, default: "/tb3_0/map"): This parameter defines the topic name on which the node will recieve the map.

#### 4.3.2. Subscribed Topics
 - The map (Topic name is defined by the ```~map_topic``` parameter) ([nav_msgs/OccupancyGrid](http://docs.ros.org/api/nav_msgs/html/msg/OccupancyGrid.html))

#### 4.3.3. Published Topics
 - ```detected_points``` ([geometry_msgs/PointStamped Message](http://docs.ros.org/api/geometry_msgs/html/msg/PointStamped.html)): The topic on which the node publishes the detected frontier points.

- ```shapes``` ([visualization_msgs/Marker Message](http://docs.ros.org/api/visualization_msgs/html/msg/Marker.html)): On this topic, the node publishes the detected points to be visualized using Rviz.

### 4.4. CQLite Filter
The filter nodes receives the detected frontier points from all the detectors, filters the points, and passes them to the assigner node to command the robots. Filteration includes the delection of old and invalid points, and it also dicards redundant points.

#### 4.4.1. Parameters
 - ```~map_topic``` (string, default: "/tb3_0/map"): This parameter defines the topic name on which the node will recieve the map. The map is used to know which points are no longer frontier points (old points).
  - ```~costmap_clearing_threshold``` (float, default: 70.0): Any frontier point that has an occupancy value greater than this threshold will be considered invalid. The occupancy value is obtained from the costmap. 
  - ```~info_radius```(float, default: 1.0): The information radius used in calculating the information gain of frontier points.
  - ```~goals_topic``` (string, default: "/detected_points"): defines the topic on which the node receives detcted frontier points.
  - ```~rate```(float, default: 100): node loop rate (in Hz).

#### 4.4.2. Subscribed Topics
 - The map (Topic name is defined by the ```~map_topic``` parameter) ([nav_msgs/OccupancyGrid](http://docs.ros.org/api/nav_msgs/html/msg/OccupancyGrid.html)).

- ```tb3_0/move_base/global_costmap/costmap``` ([nav_msgs/OccupancyGrid](http://docs.ros.org/api/nav_msgs/html/msg/OccupancyGrid.html)): where x (in robot_x) refers to robot's number. 

The filter node subscribes for all the costmap topics of all the robots, the costmap is required therefore. Normally, costmaps should be published by the navigation stack (after bringing up the navigation stack on the robots, each robot will have a costmap).

 - The goals topic (Topic name is defined by the ```~goals_topic``` parameter)([geometry_msgs/PointStamped Message](http://docs.ros.org/api/geometry_msgs/html/msg/PointStamped.html)): The topic on which the filter node receives detected frontier points.
 
#### 4.4.3. Published Topics

 - ```frontiers``` ([visualization_msgs/Marker Message](http://docs.ros.org/api/visualization_msgs/html/msg/Marker.html)): The topic on which the filter node publishes the received frontier points for visualiztion on Rviz.
 
 - ```centroids``` ([visualization_msgs/Marker Message](http://docs.ros.org/api/visualization_msgs/html/msg/Marker.html)): The topic on which the filter node publishes only the filtered frontier points for visualiztion on Rviz.

 - ```filtered_points``` ([PointArray](./msg/PointArray.msg)): All the filtered points are sent as an array of points to the assigner node on this topic.

### 4.5. CQLite Planner
This node recieve target exploration goals, which are the filtered frontier points published by the filter node, and commands the robots accordingly. The assigner node commands the robots through the ```move_base```. This is why you have bring up the navigation stack on your robots.

#### 4.5.1. Parameters
- ```~map_topic``` (string, default: "/robot_1/map"): This parameter defines the topic name on which the node will recieve the map. In the single robot case, this topic should be set to the map topic of the robot. In the multi-robot case, this topic must be set to global merged map.
 - ```~info_radius```(float, default: 1.0): The information radius used in calculating the information gain of frontier points.
  
 - ```~info_multiplier```(float, default: 3.0): The unit is meters. This parameter is used to give importance to information gain of a frontier point over the cost (expected travel distance to a frontier point).
  
- ```~hysteresis_radius```(float, default: 3.0): The unit is meters. This parameter defines the hysteresis radius.

- ```~hysteresis_gain```(float, default: 2.0): The unit is metesr. This parameter defines the hysteresis gain.
 
- ```~frontiers_topic``` (string, default: "/filtered_points"): The topic on which the assigner node receives filtered frontier points.

-  ```~delay_after_assignement```(float, default: 0.5): The unit is seconds. It defines the amount of delay after each robot assignment.

- ```~rate```(float, default: 100): node loop rate (in Hz).

#### 4.5.2. Subscribed Topics
 - The map (Topic name is defined by the ```~map_topic``` parameter) ([nav_msgs/OccupancyGrid](http://docs.ros.org/api/nav_msgs/html/msg/OccupancyGrid.html)).
 
- Filtered frontier points topic (Topic name is defined by the ```~frontiers_topic``` parameter)  ([PointArray](./msg/PointArray.msg)).

#### 4.5.3. Published Topics
The assigner node does not publish anything. It sends the assigned point to the ```move_base``` using Actionlib.

## 5. Launch
Run the CQLite package after installation on a robot and source bash and /devel/setup.bash file:

`` $ roslaunch cqlite cqlite_exploration.launch ``

## Core contributors

* **Ehsan Latif** - PhD Candidate

* **Dr. Ramviyas Parasuraman** - Lab Director


## Heterogeneous Robotics (HeRoLab)

**Heterogeneous Robotics Lab (HeRoLab), School of Computing, University of Georgia.** 

For further information, contact Ehsan Latif ehsan.latif@uga.edu or Dr. Ramviyas Parasuraman ramviyas@uga.edu

http://hero.uga.edu/

<p align="center">
<img src="http://hero.uga.edu/wp-content/uploads/2021/04/herolab_newlogo_whitebg.png" width="300">
</p>







