#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import rospy
import numpy as np
import time 

from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from extended_navigation_mapping.srv import GetDistance
from extended_mapping.map_processing import *
from extended_mapping.ros_conversions import *
from navigation_utils.agent_pos_listener import *

class AgentDistPublisher:
    def __init__(self, dist_server, agent_frame, map_frame, agent_dist_topic):
        self.dist_serv_proxy = rospy.ServiceProxy(dist_server + "/get_distance", GetDistance)
        #self.odom_subscriber = rospy.Subscriber(odom_topic, Odometry, self.odomCallback)
        self.agent_pos_listener = AgentPosListener(
            map_frame, agent_frame)
        self.agent_dist_publisher = rospy.Publisher(agent_dist_topic, Float64, queue_size=1, latch=True)

        self.curr_pos = None
        self.cached_dist_map = None

    def updateMap(self):
        self.cached_dist_map = envMapFromRosEnvGridMapMsg(self.dist_serv_proxy().distance)
        
    def publishDist(self):
        self.curr_pos = self.agent_pos_listener.get2DAgentPos()
        if self.curr_pos is not None and self.cached_dist_map is not None:
            self.agent_dist_publisher.publish(Float64(self.cached_dist_map.valueAt(self.curr_pos)))

if __name__ == "__main__":   
    rospy.init_node("agent_dist_publisher")
    r = rospy.Rate(5) 
    
    map_frame = rospy.get_param("~map_frame","map")
    agent_frame = rospy.get_param("~agent_frame","base_link")
    dist_server_node_name = rospy.get_param("~dist_server_node_name","dist_server")
    cached_dist_map = None

    ag_dist_publisher = AgentDistPublisher(dist_server_node_name, agent_frame, map_frame, "agent_dist")
    rospy.wait_for_service(dist_server_node_name + "/get_distance")
    time.sleep(1.5)

    while not rospy.is_shutdown():
        r.sleep()
        ag_dist_publisher.updateMap()
        ag_dist_publisher.publishDist()
    rospy.spin()