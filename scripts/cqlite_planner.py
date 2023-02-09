#!/usr/bin/env python
 
import sys
import math
import numpy as np
import heapq as hq
import time

######### ROS CODE 

import rospy
from geometry_msgs.msg import Twist 
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations  import euler_from_quaternion, quaternion_from_euler
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import cv2
from std_msgs.msg import Float32
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import Twist, Pose, PoseStamped
import rospy
import copy

# initialization
path = Path()
trajectory = []
x = y = theta = 0.0

def get_rotation (msg):
    global x, y, theta 
    global path
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    euler = euler_from_quaternion (orientation_list)
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    theta = euler[2]
    print('X =',round(float(x),1), 'Y =',round(float(y),1), 'theta =',round(float(theta),1))
    path_pub = rospy.Publisher('/trajectory', Path, queue_size=10)
    path.header = msg.header
    pose = PoseStamped()
    pose.header = msg.header
    pose.pose = msg.pose.pose
    path.poses.append(pose)
    trajectory.append((x,y,theta))
    path_pub.publish(path)

############# ROS CODE ENDs

SIMILARITY_THRESHOLD = 0.1
SAFETY_OFFSET = 5    # number of pixels away from the wall the robot should remain
ALGORITHM = "SARSOP-POMDP" ##POMDP, SARSOP-POMDP

## robot node
class Node:
    def __init__(self, x, y, theta=0.0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        
        # f(n) = h(n) + g(n)
        ## heurestic
        self.f = 0 ## sum of h and g
        self.h = 0 ## distance from current pose to goal
        self.g = 0 ## distance from origin to current pose
        self.resolution = 0.05  ##
        ## actual map dimmention 2000 by 2000
        ## divided grid 100 by 100

    def euclidean_distance(self, goal):
        return math.sqrt(math.pow((goal.x-self.x),2) + math.pow((goal.y-self.y),2))

    ## motion model
    def apply_move(self, move):
        theta_new = self.theta + move[1]
        x_new = self.x + math.cos(theta_new) * move[0]    # d.cos(theta)
        y_new = self.y + math.sin(theta_new) * move[0]  # d.sin(theta)
        return Node(x_new, y_new, theta_new)
    
    ## check moveable state
    def is_allowed(self,state,grid_map):
        was_error = False
        i, j = state[0],state[1]
        # side = int(math.floor((max(i, j) / self.resolution) / 2))
        side = ROBOT_SIZE
        try:
            for s_i in range(i-side, i+side):
                for s_j in range(j-side, j+side):
                    cell = grid_map[s_i][s_j]
                    if cell == 100 or cell == -1:
                        return False
        except IndexError as e:
            # rospy.loginfo("Indices are out of range")
            was_error = True
        return True and not was_error
    
    def __lt__(self, other):
             if self.f < other.f:
                  return True
             else:
                  return False
              
    def is_move_valid(self, grid_map, move):
        goal = self.apply_move(move)
        # convert goal coordinates to pixel coordinates before checking this
        goal_pixel = self.world_to_pixel((goal.x, goal.y), (-50, -50),self.resolution)
        # check if too close to the walls
        # if not is_allowed(goal_pixel,grid_map):
        #     return false
        # if (goal_pixel[0] <= SAFETY_OFFSET or goal_pixel[1] <= SAFETY_OFFSET)  and not is_allowed(goal_pixel,grid_map):#grid_map[goal_pixel[0]-SAFETY_OFFSET][goal_pixel[1]]:
        #     return False
        # # if goal_pixel[1] >= SAFETY_OFFSET and not grid_map[goal_pixel[0]][goal_pixel[1]-SAFETY_OFFSET]:
        # #     return False
        # if goal_pixel[0] >= SAFETY_OFFSET and goal_pixel[1] >= SAFETY_OFFSET and not grid_map[goal_pixel[0]-SAFETY_OFFSET][goal_pixel[1]-SAFETY_OFFSET]:
        #     return False
        # if grid_map[goal_pixel[0]][goal_pixel[1]]:
        #     return True
        return self.is_allowed(goal_pixel,grid_map)
    
    def world_to_pixel(self,pos,origin,resolution):
        pixel_points = [0,0]
        pixel_points[0] = int((pos[0] - origin[0]) / resolution)
        pixel_points[1] = int((pos[1] - origin[1]) / resolution)
        return pixel_points

    def is_valid(self, grid_map):
        """
        Return true if the location on the map is valid, ie, in obstacle free zone
        """
        goal_pixel = self.world_to_pixel((self.x, self.y),(-50, -50),0.05)
        if grid_map[goal_pixel[0]][goal_pixel[1]] != -1:
            return True
        return False

    def is_similar(self, other):
        """
        Return true if other node is in similar position as current node
        """
        return self.euclidean_distance(other) <= SIMILARITY_THRESHOLD
def radians(degree):
    return (degree * math.pi / 180)




G_MULTIPLIER = 0.2

MOVES = [ (0.2, radians(0)),     # move ahead 
          (-0.2, radians(0)),     # move backwards 
          (0, radians(90)),     # turn left 
          (0, -radians(90)) ]    # turn right 
            
TOLERANCE = 0.2
ROBOT_SIZE = 10

TRANSITION_PROBABILITY = [ 0.8, 0.066, 0.066, 0.066]
OBSERVATION_PROBABILITY = 0.8
SAMPLE_SIZE = 1000
MOVE_RANGE = 3


##### ROS CODE

def move_robot(x,y,w=1):

    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        return client.get_result()

############# ROS CODE END

class PathPlanner:
    def __init__(self, start, theta, goal,grid):
        #print("building map....")
        # map remains constant
        self.map = grid
        self.start = start
        self.theta = theta
        self.goal = goal
        #print(np.shape(grid))
        #print("map built. planner initialized")
    def bestPathsHandler(self,paths, node):
        if len(paths) < 10:
            paths.append(node)
        else:
            index = 0
            longestPathNode = paths[0]
            for i in range(len(paths)):
                if paths[i].f > longestPathNode.f:
                    longestPathNode = paths[i]
                    index = i
            if node.f < longestPathNode.f:
                paths[index] = node 
        return paths

    def subPomdpHandler(self,start,end,grid_map):
        opened = []
        closed = []     
        bestNode = None
        MOVE_RANGE = 3
        final = None
        finalBool = False
        hq.heappush(opened, (0.0, start))
        while opened:
            q = hq.heappop(opened)[1]
            for move in MOVES:
                if (q.is_move_valid(grid_map, move)):
                    next_node = q.apply_move(move)    # Node is returned in world coordinates
                else:
                    next_node = None
                    if bestNode == None:
                        bestNode = q
                    elif q.f < bestNode.f:
                        bestNode = q
                if next_node != None:
                    
                    next_node.h = next_node.euclidean_distance(end) #h - dist from node to end
                    next_node.g = q.g + next_node.euclidean_distance(q) #g - dist from origin to node
                    next_node.f = G_MULTIPLIER * next_node.g + next_node.h
                    
                    #print("next g", next_node.g)
                    if next_node.g > MOVE_RANGE: #out of sensor bounds
                        continue
                    if q.f == math.inf:
                        next_node.parent = start
                    else:
                        next_node.parent = q

                    if next_node.euclidean_distance(end) < TOLERANCE:
                        if final == None:
                            final = next_node
                        elif next_node.f < final.f:
                            final = next_node                  
                        continue

                    if bestNode == None:
                        bestNode = next_node
                    elif next_node.f < bestNode.f:
                        bestNode = next_node
                    
                    potential_open = any(other_f <= next_node.f and other_next.is_similar(next_node) for other_f, other_next in opened)
                    if not potential_open:
                        potential_closed = any(other_next.is_similar(next_node) and other_next.f <= next_node.f for other_next in closed)
                        if not potential_closed:
                            hq.heappush(opened, (next_node.f, next_node)) #push to opened
                    
            closed.append(q)
        if final:
            print("sub final found")
            bestNode = final
            finalBool = True
        return bestNode, finalBool

    def pomdpHandler(self,start, end, grid_map):
        #print("start", start.x, start.y)
        #print("end", end.x, end.y)

        bestSubNodes = []
        opened = []
        closed = []     
        bestNode = None
        MOVE_RANGE = 3
        final = None
        finalBool = False
        bestPaths = []
        hq.heappush(opened, (0.0, start))
        while opened:
            q = hq.heappop(opened)[1]
            for move in MOVES:
                if (q.is_move_valid(grid_map, move)):
                    next_node = q.apply_move(move)    # Node is returned in world coordinates
                else:
                    next_node = None
                    if bestNode == None:
                        bestNode = q
                        bestPaths = self.bestPathsHandler(bestPaths,q)
                    elif q.f < bestNode.f:
                        bestNode = q
                        bestPaths = self.bestPathsHandler(bestPaths,q)
                if next_node != None:
                    
                    next_node.h = next_node.euclidean_distance(end) #h - dist from node to end
                    next_node.g = q.g + next_node.euclidean_distance(q) #g - dist from origin to node
                    next_node.f = G_MULTIPLIER * next_node.g + next_node.h
                    
                    #print("next g", next_node.g)
                    if next_node.g > MOVE_RANGE: #out of sensor bounds
                        continue
                    if q.f == math.inf:
                        next_node.parent = start
                    else:
                        next_node.parent = q

                    if next_node.euclidean_distance(end) < TOLERANCE:
                        if final == None:
                            final = next_node
                        elif next_node.f < final.f:
                            final = next_node                  
                        continue

                    if bestNode == None:
                        bestNode = next_node
                        bestPaths = self.bestPathsHandler(bestPaths,q)
                    elif next_node.f < bestNode.f:
                        bestNode = next_node
                        bestPaths = self.bestPathsHandler(bestPaths,q)
                    
                    potential_open = any(other_f <= next_node.f and other_next.is_similar(next_node) for other_f, other_next in opened)
                    if not potential_open:
                        potential_closed = any(other_next.is_similar(next_node) and other_next.f <= next_node.f for other_next in closed)
                        if not potential_closed:
                            hq.heappush(opened, (next_node.f, next_node)) #push to opened
                    
            closed.append(q)

        if final:
            print("primary final found")
            bestNode = final
            bestSubNodes = []
            finalBool = True
            return bestNode, finalBool, bestSubNodes

        for path in bestPaths:
            path.f = math.inf
            path.g = 0
            tempNode, subFinalDestReached = self.subPomdpHandler(path, end, grid_map)
            while not subFinalDestReached:
                tempNode.f = math.inf
                tempNode.g = 0
                tempNode, subFinalDestReached = self.subPomdpHandler(tempNode, end, grid_map)
            if tempNode:
                bestSubNodes.append(tempNode)

        return bestNode, finalBool, bestSubNodes

        

    def pomdp(self, start, end, grid_map):
        possibleSubPaths = []
        if not end.is_valid(grid_map):
            print("goal invalid")
            return None
        print("goal valid")
        node, finalDestReached, bestSubNodes = self.pomdpHandler(start, end, grid_map)
        for item in bestSubNodes:
                possibleSubPaths.append(bestSubNodes)
        while not finalDestReached:
            node.f = math.inf
            node.g = 0
            node, finalDestReached, bestSubNodes = self.pomdpHandler(node, end, grid_map)
            for item in bestSubNodes:
                possibleSubPaths.append(bestSubNodes)
        # print("Printing all possible sub path nodes")
        # if possibleSubPaths:
        #     for item in possibleSubPaths:
        #         print(item)

        
        #node, finalDestReached = self.pomdpHandler(node, end, grid_map)
        
        # tempNode = node
        # while tempNode:
        #     print(tempNode.x, tempNode.y)
        #     tempNode = tempNode.parent
        return node #, possibleSubPaths //contains every possible "red" line
    
    def sarshop_pomdpHandler(self,start, end, grid_map):      
        opened = []
        closed = []            
        
        final = None
 
        hq.heappush(opened, (0.0, start))
        while (final == None) and opened:
            q = hq.heappop(opened)[1]
            for move in MOVES:
                if (q.is_move_valid(grid_map, move)):
                    next_node = q.apply_move(move)    # Node is returned in world coordinates
                else:
                    next_node = None
                    
                if next_node != None:
                    
                    ## computer reward
                    reward = 0
                    for move in MOVES:
                            nnd = q.apply_move(move)
                            nnd.h = nnd.euclidean_distance(end) #h - dist from node to end
                            nnd.g = q.g + nnd.euclidean_distance(q) #g - dist from origin to node
                            if nnd == next_node:
                                nnd.f = (G_MULTIPLIER * nnd.g + nnd.h) * TRANSITION_PROBABILITY[0]
                            else:
                                nnd.f = (G_MULTIPLIER * nnd.g + nnd.h) * TRANSITION_PROBABILITY[1]
                            reward += nnd.f
                    next_node.g = q.g + next_node.euclidean_distance(q) * TRANSITION_PROBABILITY[0]
                    next_node.h = next_node.euclidean_distance(end) * TRANSITION_PROBABILITY[0]
                    next_node.f = reward
                    
                    #print("next g", next_node.g)
                    if next_node.g > MOVE_RANGE: #out of sensor bounds
                        continue
                    if q.f == math.inf:
                        next_node.parent = start
                    else:
                        next_node.parent = q

                    if next_node.euclidean_distance(end) < TOLERANCE:
                        if final == None:
                            final = next_node
                        elif next_node.f < final.f:
                            final = next_node                  
                        break
                    
                    potential_open = any(other_f <= next_node.f and other_next.is_similar(next_node) for other_f, other_next in opened)
                    if not potential_open:
                        potential_closed = any(other_next.is_similar(next_node) and other_next.f <= next_node.f for other_next in closed)
                        if not potential_closed and len(opened) <= SAMPLE_SIZE:
                            hq.heappush(opened, (next_node.f, next_node)) #push to opened

            closed.append(q)

        return final
    
    
    def sarsop_pomdp(self, start, end, grid_map):
        if not end.is_valid(grid_map):
            print("goal invalid")
            return None
        print("goal valid")
        final = self.sarshop_pomdpHandler(start, end, grid_map)
        return final
        
    def a_star(self,start, end, grid_map):

        if not end.is_valid(grid_map):
            print("goal invalid")
            return None
        print("goal valid")
        opened = []
        closed=[]
        final = None
        hq.heappush(opened, (0.0, start))

        while (final == None) and opened:
            # q is a Node object with x, y, theta
            q = hq.heappop(opened)[1]
            for move in MOVES:        # move is in world coordinates
                if (q.is_move_valid(grid_map, move)):
                    next_node = q.apply_move(move)    # Node is returned in world coordinates
                else:
                    next_node = None
                #print("next node is : ", next_node) 
                if next_node != None:
                    if next_node.euclidean_distance(end) < TOLERANCE:
                        next_node.parent = q                    
                        final = next_node
                        break
                    # update heuristics h(n) and g(n)
                    next_node.h = next_node.euclidean_distance(end)
                    next_node.g = q.g + next_node.euclidean_distance(q)
                    # f(n) = h(n) + g(n)
                    next_node.f = G_MULTIPLIER * next_node.g + next_node.h
                    next_node.parent = q

                    # other candidate locations to put in the heap
                    potential_open = any(other_f <= next_node.f and other_next.is_similar(next_node) for other_f, other_next in opened)
                    
                    if not potential_open:
                        potential_closed = any(other_next.is_similar(next_node) and other_next.f <= next_node.f for other_next in closed)
                        if not potential_closed:
                            hq.heappush(opened, (next_node.f, next_node))
            closed.append(q)    

        return final   

    def plan(self):
        start = time.time()
        if ALGORITHM == "A*":
            final = self.a_star(self.start, self.goal, self.map)
        elif ALGORITHM == "POMDP":
             final = self.a_star(self.start, self.goal, self.map) ## replace with POMDP
        elif ALGORITHM == "SARSOP-POMDP":
             final = self.a_star(self.start, self.goal, self.map) ## replace with SARSOP-POMDP
        else:
            print("No algorithm selected")
            
        end = time.time()
        print("--- Computation time: %sseconds ---" % (end - start))
        if final == None:
            print("Path not found.")
        else:
            print("Constructing path..")
            path = self.construct_path(final)    # path in world coordinates
            print("path: ")
            points = []
            for step in path:
                points.append((step.x, step.y,step.theta))
            # publish this path - safegoto for each of the path components
            points.reverse()
            points = points[1:]
            points.append((self.goal.x, self.goal.y,self.goal.theta))
            for p in range(len(points)):
                print("x:", points[p][0], " y:", points[p][1]," theta:", points[p][2])
            np.savetxt('path.dat', points,fmt='%.2f',delimiter='\t')

            # first process the points
            translate_x = points[0][0]
            translate_y = points[0][1]
            for p in range(len(points)):
                new_x = points[p][0] - translate_x
                new_y = points[p][1] - translate_y
                if self.theta == math.pi/2:
                    points[p] = [-new_y, new_x,points[p][2]]
                elif self.theta == math.pi:
                    points[p] = [-new_x, -new_y,points[p][2]]
                elif self.theta == -math.pi/2:
                    points[p] = [new_y, -new_x,points[p][2]]
                else:            
                    points[p] = [new_x, new_y,points[p][2]]
            # translate coordinates for theta            
            
                
            # run safegoto on the translated coordinates
            # start = time.time()
            # robot = SafeGoTo()
            # robot.travel(points)
            # end = time.time()
            # print("--- Execution time: %s seconds ---" % (start - end))
            return True


    def construct_path(self, end):
        """
        backtrack from end to construct path
        """
        current = end
        path = []    # path needs to be in world coordinates
        while current != None:
            path.append(current)
            current = current.parent
        return path



#### ROS CODE START

LINEAR_VELOCITY = 0.2
ANGULAR_VELOCITY = 0.4
TOLERANCE = 0.3
ROBOT_RADIUS = 0.22
OBSTACLE_THRESHOLD = 0.78
EXIT_STATUS_ERROR = 1
EXIT_STATUS_OK = 0

class SafeGoTo:
    def __init__(self):
        # rospy.init_node('traveler', anonymous=True)
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # Hold position and quaternion of robot            
        self.pos = Pose()
        self.theta = math.pi
        self.obstacle_found = False
        self.obstacle_circumventing = False
        self.start = (0, 0)
        self.goal = None
        self.mline = None
        self.curr_line = None
        self.sonar_data = []
        self.vel_msg = Twist()
    

    def slope(self, p1, p2):
        delta_y = p2[1]-p1[1]
        delta_x = p2[0]-p1[0]
        return delta_y/delta_x if delta_x!=0 else float('inf')

    def euclidean_distance(self):
        return math.sqrt(math.pow((self.goal[0]-self.pos.position.x),2) + math.pow((self.goal[1]-self.pos.position.y),2))

    def angular_difference(self):
        return math.atan2(self.goal[1]-self.pos.position.y, self.goal[0]-self.pos.position.x) - self.theta

    
    def stop(self):

        self.vel_msg.linear.x = 0
        self.vel_msg.angular.z = 0
        self.vel_publisher.publish(self.vel_msg)    


    def go(self):
        
        # keep traveling until distance from current position to goal is greater than 0        
        while self.euclidean_distance() > TOLERANCE and not self.obstacle_found:
            #print("distance from goal " + str(self.goal) + ": ", self.euclidean_distance())
            # set linear velocity            
            self.vel_msg.linear.x = min(LINEAR_VELOCITY, 
                                        LINEAR_VELOCITY * self.euclidean_distance())
            self.vel_msg.linear.y = 0
            self.vel_msg.linear.z = 0
            # set angular velocity
            self.vel_msg.angular.x = 0
            self.vel_msg.angular.y = 0
            self.vel_msg.angular.z = min(ANGULAR_VELOCITY, 
                                         ANGULAR_VELOCITY * self.angular_difference())
            # publish velocity
            self.vel_publisher.publish(self.vel_msg)
        
        if self.obstacle_found:
            self.stop()            
            # self.bug()    
        else:
            self.stop()
            self.start = self.goal

    def travel(self, goals):
        for goal in goals:
            self.start = (self.pos.position.x, self.pos.position.y)            
            self.mline = self.slope(self.start, goal)
            self.goal = goal            
            # self.go()
            move_robot(self.goal[0],self.goal[1],self.goal[2])
            message = "Position " + str(self.goal) + " has been achieved."
            rospy.loginfo(message)




map_grid = []
def grid_callback(data):
    global map_grid
    map_grid = np.array(data.data)
    # print(data.info)
    map_grid = map_grid.reshape(data.info.width,data.info.height,order='F')
    # np.savetxt('map', map_grid)
    print("In subscription: ",np.shape(map_grid))

############# ROS CODE END

destination = [(5,-3),(7.5,7.5)]#,(7.5,-7.5),(-7.5,-7.5),(-7.5,7.5),(0,0)]

def main():
    global vel
    global map_grid
    #Initialize our node
    rospy.init_node("cqlite_planner")
    map_topic= rospy.get_param('~map_topic','/map')
	info_radius= rospy.get_param('~info_radius',1.0)					#this can be smaller than the laser scanner range, >> smaller >>less computation time>> too small is not good, info gain won't be accurate
	info_multiplier=rospy.get_param('~info_multiplier',3.0)		
	hysteresis_radius=rospy.get_param('~hysteresis_radius',3.0)			#at least as much as the laser scanner range
	hysteresis_gain=rospy.get_param('~hysteresis_gain',2.0)				#bigger than 1 (biase robot to continue exploring current region
	frontiers_topic= rospy.get_param('~frontiers_topic','/filtered_points')	
	n_robots = rospy.get_param('~n_robots',2)
	namespace = rospy.get_param('~namespace','')
	namespace_init_count = rospy.get_param('namespace_init_count',0)
	delay_after_assignement=rospy.get_param('~delay_after_assignement',0.5)
	rateHz = rospy.get_param('~rate',100)


    rate = rospy.Rate(rateHz)
	rospy.Subscriber(map_topic, OccupancyGrid, mapCallBack)
	rospy.Subscriber(frontiers_topic, PointArray, callBack)		
	while len(frontiers)<1:
		pass
	centroids=copy(frontiers)	
	while (len(mapData.data)<1):
		pass

	robots=[]
	if len(namespace)>0:
		for i in range(0,n_robots):
			robots.append(robot(namespace+str(i+namespace_init_count)))
	elif len(namespace)==0:
			robots.append(robot(namespace))
	for i in range(0,n_robots):
		robots[i].sendGoal(robots[i].getPosition())

    rospy.Subscriber("/map", OccupancyGrid, grid_callback)

    r = rospy.Rate(10)
    # start = Node(0,0,math.pi)
    # goal = Node(destination[0][0], destination[0][1])
    # delt = 0
    # while len(map_grid) == 0:
    #     delt = delt + 1
    # planner = PathPlanner(start, math.pi, goal,map_grid)
    # path = planner.plan()
    # if path == True:
    #     print("Arrived at Goal!")
    for pos in destination:
        print(pos)
        result_move = move_robot(pos[0],pos[1])
        delt = 0
        while len(map_grid) == 0:
            delt = delt + 1
        if result_move:
            print("Arrived at Goal: "+str(pos))
    np.savetxt('path_followed.dat', trajectory,fmt='%.2f',delimiter='\t')

    rospy.spin()


if __name__ == "__main__":
    # try:
        main()
    # except rospy.ROSInterruptException:
    #     pass
