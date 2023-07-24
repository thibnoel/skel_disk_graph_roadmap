#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import numpy as np
import time
import rospy
import actionlib
from extended_mapping.map_processing import *
from extended_mapping.ros_conversions import *
from sdg_roadmap.skel_disk_graph_provider import *

from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_msgs.msg import Int32, Float32, Bool
from std_srvs.srv import Empty
from geometry_msgs.msg import Point
from nav_msgs.srv import GetMap
from extended_navigation_mapping.msg import FollowPathAction, FollowPathGoal, FollowPathActionResult
from extended_navigation_mapping.srv import GetDistance
from skeleton_disk_graph_roadmap.srv import GetDiskGraph

from sdg_roadmap_server import SDGRoadmapServer
from sdg_exploration_visualizer import SDGExplorationVisualizer


class SDGExplorationServer:
    def __init__(self, occ_map_service, path_following_node_name, dist_server_node_name, map_frame, agent_frame, sdg_tuning_param, sdg_strat_param, safety_dist):
        # TO PARAMETRIZE SOMEWHERE ELSE
        self.path_cost_param = {
            "energy_penalty": -1,
            "coverage_reward": 1
        }
        # Subscribers
        self.pf_result_subscriber = rospy.Subscriber(
            pf_node_name + "/follow_path/result", FollowPathActionResult, self.pfResultCallback)
        # Publishers
        self.cancel_publisher = rospy.Publisher(
            pf_node_name + "/follow_path/cancel", GoalID, queue_size=1)
        # Action server client
        self.follow_path_client = actionlib.SimpleActionClient(
            pf_node_name+"/follow_path", FollowPathAction)
        # Service proxy
        self.occ_map_proxy = rospy.ServiceProxy(occ_map_service, GetMap)
        self.dist_map_proxy = rospy.ServiceProxy(dist_server_node_name + "/get_distance", GetDistance)
        
        # Skeleton Disk-Graph Roadmap
        self.sdg_roadmap_server = SDGRoadmapServer(dist_server_node_name, map_frame, agent_frame, sdg_tuning_param)
        self.sdg_exploration_path_provider = SDGExplorationPathProvider(
            self.sdg_roadmap_server.sdg_provider,
            sdg_strat_param["unkn_occ_range"],
            sdg_strat_param["frontiers_max_known_ratio"],
            sdg_strat_param["search_dist_increment"],
            self.path_cost_param
        )

        self.safety_dist = safety_dist
        self.visualizer = SDGExplorationVisualizer()
        # State
        self.is_following_path = False
        self.current_target_pos = None
        self.current_path = None
        self.current_goal = None
        self.replan_pos = []
        self.past_plans = []
        

    def resetPlanner(self, sdg_tuning_param_dict):
        rospy.logwarn("RESETTING PLANNER")
        self.sdg_roadmap_server.resetPlanner(sdg_tuning_param_dict)
    
    def pfResultCallback(self, pfResultMsg):
        rospy.loginfo(str(pfResultMsg))
        self.active_target = None
        self.is_following_path = False

    def callPathFollowingDetached(self, goal_path):
        #self.current_path = goal_path
        rospy.loginfo("Start path following")
        self.current_goal = self.follow_path_client.send_goal(goal_path)
        self.is_following_path = True

    def checkPathValidity(self):
        if self.current_target_pos is not None :
            dist_map = envMapFromRosEnvGridMapMsg(self.dist_map_proxy().distance)
            for k, p in enumerate(self.current_path.waypoints):
                d = dist_map.valueAt(p)
                checkColl = d > self.safety_dist
                if not checkColl:
                    rospy.loginfo("INTERRUPTING INVALID PATH")
                    cancel_msg = GoalID()
                    self.cancel_publisher.publish(cancel_msg)
        #return True

    def selectPath(self):
        start = self.sdg_roadmap_server.agent_pos_listener.get2DAgentPos()
        dist_map = envMapFromRosEnvGridMapMsg(self.dist_map_proxy().distance)
        self.replan_pos.append(start)
        self.sdg_roadmap_server.updatePlanner(None)
        frontiers_paths = self.sdg_exploration_path_provider.getFrontiersPaths(start, envMapFromRosEnvGridMapMsg(self.occ_map_proxy().map))
        if not len(frontiers_paths):
            rospy.logwarn("No frontiers found")
            return None
        best_id, best_path = self.sdg_exploration_path_provider.selectExplorationPath(frontiers_paths)
        if best_path is None:
            return None
        self.current_target_pos = best_path['path'].waypoints[-1]
        best_path = best_path['path']
        #path_subdiv_length = 0.5*np.min(raw_best_path.radii)
        #simplified_path = raw_best_path.getSubdivized(int(raw_best_path.getTotalLength()/path_subdiv_length), dist_map).getReducedBubbles()
        #spline_path = simplified_path.getSmoothedSpline(dist_map)
        self.current_path = best_path
        self.past_plans.append(best_path.waypoints)
        self.visualizer.publishReplanPosViz(self.replan_pos)
        self.visualizer.publishFrontiersPlanViz(frontiers_paths, best_id)
        self.visualizer.publishPastPlanViz(self.past_plans[:-1])

        return self.sdg_roadmap_server.constructPathMsg(best_path.waypoints)

    def interruptOnInvalidTarget(self):
        occ_map = envMapFromRosEnvGridMapMsg(self.occ_map_proxy().map)
        dist_map = envMapFromRosEnvGridMapMsg(self.dist_map_proxy().distance)
        if self.current_target_pos is not None :
            target_rad = dist_map.valueAt(self.current_target_pos)
            if target_rad < self.sdg_roadmap_server.sdg_provider.parameters_dict["min_pbubbles_rad"] :
                cancel_msg = GoalID()
                rospy.loginfo("INTERRUPTING INVALID TARGET RAD")
                self.cancel_publisher.publish(cancel_msg)
                return
            
            # Interrrupt if target becomes known
            if not self.sdg_exploration_path_provider.checkFrontierValidity(self.current_target_pos, occ_map, dist_map):
                self.current_target_id = None
                rospy.loginfo("INTERRUPTING TARGET BECAME KNOWN")
                cancel_msg = GoalID()
                self.cancel_publisher.publish(cancel_msg)

    def updateCurrBubbleViz(self):
        curr_pos = self.sdg_roadmap_server.agent_pos_listener.get2DAgentPos()
        curr_max_bub = self.sdg_roadmap_server.sdg_provider.biggestContainingBubble(curr_pos)
        if curr_max_bub is not None:
            neighb = [self.sdg_roadmap_server.sdg_provider.graph.nodes[i]["node_obj"] for i in list(self.sdg_roadmap_server.sdg_provider.graph.neighbors(curr_max_bub.id))]
            b_pos = [n.pos for n in neighb] + [curr_max_bub.pos]
            b_rad = [n.bubble_rad for n in neighb] + [curr_max_bub.bubble_rad]
        
            self.visualizer.publishCurrBubbleViz(b_pos, b_rad)


if __name__ == "__main__":
    rospy.init_node("exploration_server")

    occ_map_service = rospy.get_param("~occ_map_service", "/rtabmap/get_map")
    pf_node_name = rospy.get_param("~path_following_node_name", "path_follower")
    
    dist_server_node_name = rospy.get_param("~dist_server_node_name","dist_server")
    map_frame = rospy.get_param("~map_frame","map")
    agent_frame = rospy.get_param("~agent_frame","base_link")
    safety_distance = rospy.get_param("~safety_distance", 0.05)

    sdg_tuning_param_dict = {
        "min_jbubbles_rad" : rospy.get_param("~sdg/min_jbubbles_rad",0.8),
        "min_pbubbles_rad" : rospy.get_param("~sdg/min_pbubbles_rad",0.4),
        "bubbles_dist_offset" : rospy.get_param("~sdg/bubbles_dist_offset",0.1),
        "knn_checks" : rospy.get_param("~sdg/knn_checks",40),
        "path_subdiv_length" : rospy.get_param("~sdg/path_subdiv_length",0.4)
    }
    sdg_strategy_param_dict = {
        "unkn_occ_range" : rospy.get_param("~strategy/unkn_occ_range",[-1,-1]),
        "frontiers_max_known_ratio" : rospy.get_param("~strategy/frontiers_max_known_ratio", 0.5),
        "search_dist_increment" : rospy.get_param("~strategy/search_dist_increment", 2)
    }

    exploration_server = SDGExplorationServer(occ_map_service, pf_node_name, dist_server_node_name, map_frame, agent_frame, sdg_tuning_param_dict, sdg_strategy_param_dict, safety_distance)
    rospy.wait_for_service(dist_server_node_name + "/get_distance_skeleton")
    rospy.wait_for_service(dist_server_node_name + "/get_distance")
    exploration_server.follow_path_client.wait_for_server()

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        r.sleep()
        if not exploration_server.is_following_path:
            
            path = exploration_server.selectPath()
            if path is not None:
                exploration_server.callPathFollowingDetached(path)
            else:
                exploration_server.resetPlanner(sdg_tuning_param_dict)
                #exit()
        else:
            exploration_server.checkPathValidity()
            exploration_server.interruptOnInvalidTarget()
        exploration_server.updateCurrBubbleViz()
    rospy.spin()