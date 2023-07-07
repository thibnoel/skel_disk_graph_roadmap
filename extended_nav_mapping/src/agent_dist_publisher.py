#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import rospy
import numpy as np
import time 

from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from ros_explore_mapping.srv import GetDistance
from map_processing.map_processing_utils import *
from nav_utilities import agent_pos_listener

class AgentDistPublisher:
    def __init__(self, dist_server, agent_frame, map_frame, agent_dist_topic):
        self.dist_serv_proxy = rospy.ServiceProxy(dist_server + "/get_distance", GetDistance)
        #self.odom_subscriber = rospy.Subscriber(odom_topic, Odometry, self.odomCallback)
        self.agent_pos_listener = agent_pos_listener.AgentPosListener(
            map_frame, agent_frame)
        self.agent_dist_publisher = rospy.Publisher(agent_dist_topic, Float64, queue_size=1, latch=True)

        self.curr_pos = None
        self.cached_dist_map = None

    #def odomCallback(self, odom_msg):
    #    self.curr_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])

    def updateMap(self):
        self.cached_dist_map = EnvironmentMap.initFromRosEnvGridMapMsg(self.dist_serv_proxy().distance)
        
    def publishDist(self):
        self.curr_pos = self.agent_pos_listener.get2DAgentPos()
        if self.curr_pos is not None and self.cached_dist_map is not None:
            self.agent_dist_publisher.publish(Float64(self.cached_dist_map.valueAt(self.curr_pos)))

if __name__ == "__main__":   
    rospy.init_node("agent_dist_publisher")
    r = rospy.Rate(1) 
    #path_follower = FollowPathServer(FollowPathServer.parseParamsFromServer(), r)
    
    map_frame = rospy.get_param("~map_frame","map")
    #agent_frame = "corrected_base_link"
    agent_frame = rospy.get_param("~agent_frame","base_link")
    dist_server_node_name = rospy.get_param("~dist_server_node_name","dist_server")
    #odom_topic = rospy.get_param("~odom_topic", "/unity_odom/odom")
    cached_dist_map = None

    ag_dist_publisher = AgentDistPublisher(dist_server_node_name, agent_frame, map_frame, "agent_dist")
    rospy.wait_for_service(dist_server_node_name + "/get_distance")
    time.sleep(1.5)

    while not rospy.is_shutdown():
        r.sleep()
        ag_dist_publisher.updateMap()
        ag_dist_publisher.publishDist()
    rospy.spin()