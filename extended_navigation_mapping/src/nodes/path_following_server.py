#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

from navigation_utils.path_following_controller import *
from navigation_utils.agent_pos_listener import *
from navigation_utils.paths import *

import rospy
import actionlib
import numpy as np

from std_msgs.msg import Bool, Header, Float64, String
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, Vector3, Pose, PoseStamped, Point, Quaternion
from extended_navigation_mapping.msg import FollowPathAction, FollowPathResult, FollowPathFeedback

class FollowPathServer :
    @staticmethod
    def parseParamsFromServer():
        # TF frames
        agent_frame = rospy.get_param("~frames/agent_frame", "base_link")
        map_frame = rospy.get_param("~frames/map_frame", "map")
        # Subscribed topics
        agent_dist_topic = rospy.get_param("~sub_topics/agent_obst_dist", "agent_dist")
        # Published topics
        vel_topic = rospy.get_param("~pub_topics/vel_topic", "/cmd_vel")
        # Provided services
        path_follower_action_server = rospy.get_param("~services/path_follower_action_server", "~follow_path")
        # Control parameters
        desired_linear_vel = rospy.get_param("~controller/desired_linear_vel", 0.2)
        max_angular_vel = rospy.get_param("~controller/max_angular_vel", 0.4)
        control_k2 = rospy.get_param("~controller/k2", 5)
        control_k3 = rospy.get_param("~controller/k3", 3)
        agent_rad = rospy.get_param("~controller/agent_radius", 0.2)
        success_goal_dist = rospy.get_param("~controller/success_goal_dist", 0.1)

        param_dict = {
                "map_frame":map_frame,
                "agent_frame":agent_frame,
                "agent_dist_topic":agent_dist_topic,
                "vel_topic":vel_topic,
                "path_follower_action_server":path_follower_action_server,
                "desired_lin_vel":desired_linear_vel,
                "max_ang_vel":max_angular_vel,
                "k2":control_k2,
                "k3":control_k3,
                "agent_radius":agent_rad,
                "success_goal_dist":success_goal_dist
            }
        return param_dict
    
    def initFromParamsDict(self, params_dict):
        self.agent_pos_listener = AgentPosListener(params_dict["map_frame"], params_dict["agent_frame"])

        self.success_goal_dist = params_dict["success_goal_dist"]
        v_des = params_dict["desired_lin_vel"]
        w_max = params_dict["max_ang_vel"]
        k2 = params_dict["k2"]
        k3 = params_dict["k3"]
        self.controller = pathFollowingController(v_des,w_max,k2,k3)

        #self.follow_path_server = actionlib.SimpleActionServer(params_dict["path_follower_action_server"], FollowPathAction, execute_cb=self.follow_path_callback, auto_start=False)
        self.follow_path_server = actionlib.SimpleActionServer("~follow_path", FollowPathAction, execute_cb=self.follow_path_callback, auto_start=False)
        self.follow_path_server.start()

        self.agent_dist_subscriber = rospy.Subscriber(params_dict["agent_dist_topic"], Float64, self.agentDistCallback)
        self.des_lin_vel_subscriber = rospy.Subscriber("~set_des_lin_vel", Float64, self.desiredLinVelCallback)
        self.des_rot_vel_subscriber = rospy.Subscriber("~set_max_rot_vel", Float64, self.desiredRotVelCallback)

        self.vel_publisher = rospy.Publisher(params_dict["vel_topic"], Twist, queue_size=1)
        #self.dist_to_path_publisher = rospy.Publisher("~dist_to_path", Float64, queue_size=1)
        #self.dist_to_goal_publisher = rospy.Publisher(dist_to_goal_topic, Point, queue_size=1)
        self.current_path = None
        self.agent_obst_dist = 0
        self.agent_radius = params_dict["agent_radius"]
    
    def __init__(self, params_dict, rate):
        self.initFromParamsDict(params_dict)
        self.rate = rate

    def desiredLinVelCallback(self, msg):
        self.controller.des_lin_vel = msg.data
    
    def desiredRotVelCallback(self, msg):
        self.controller.max_ang_vel = msg.data
    

    def agentDistCallback(self, agent_dist):
        self.agent_obst_dist = agent_dist.data

    def follow_path_callback(self, goal_path):
        waypoints = np.array([[p.pose.position.x, p.pose.position.y] for p in goal_path.path.poses])
        path = WaypointsPath(waypoints)
        success = False 

        while not success : 
            self.rate.sleep()    
            # Check for preemption
            if self.follow_path_server.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % "follow_path")
                self.follow_path_server.set_preempted()
                break

            agent_pos_rot = self.agent_pos_listener.get2DAgentPosRot()
            agent_pos = np.array(agent_pos_rot[:2])
            agent_rot = agent_pos_rot[2]
            success = np.linalg.norm(agent_pos - path.waypoints[-1]) < self.success_goal_dist
            raw_command = self.controller.computeRawCommand(path, agent_pos, agent_rot)
            command = self.controller.computeAdjustedCommand(raw_command, self.agent_obst_dist, self.agent_radius)
            if ((command[0] < 1e-6) and (np.abs(command[1]) < 1e-6)) :
                break
            #self.dist_to_path_publisher.publish(Float64(self.controller.dist_error))

            vel_command = Twist()
            vel_command.linear.x = command[0]
            vel_command.angular.z = command[1]
            self.last_command_vel = command[0]
            self.last_command_ang = command[1]
            self.vel_publisher.publish(vel_command)

            feedback = FollowPathFeedback()
            path_witness_point, wp_t = path.getNearestPoint(agent_pos)
            feedback.progress = wp_t
            self.follow_path_server.publish_feedback(feedback)
            
        vel_command = Twist()
        self.vel_publisher.publish(vel_command)
        if success :
            result = FollowPathResult()
            header = Header()
            header.stamp = rospy.Time(0)
            header.frame_id = self.agent_pos_listener.map_frame
            print("Path following succeeded")
            result.final_pose = PoseStamped(header, Pose(Point(agent_pos[0],agent_pos[1],0), Quaternion(0,0,0,1)))
            result.status = String("SUCCEEDED")
            self.follow_path_server.set_succeeded(result) 
        else :
            print("Path following interrupted")


if __name__ == "__main__":   
    rospy.init_node("path_follower")
    r = rospy.Rate(20) 
    path_follower = FollowPathServer(FollowPathServer.parseParamsFromServer(), r)
    
    while not rospy.is_shutdown():
        r.sleep()
    rospy.spin()