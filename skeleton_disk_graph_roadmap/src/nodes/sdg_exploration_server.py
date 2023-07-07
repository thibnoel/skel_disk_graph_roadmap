#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import numpy as np
import time
import rospy
from sdg_roadmap.sdg_roadmap_utils import circularMask
from sdg_roadmap.sdg_roadmap_strategy import *
from map_processing.map_processing_utils import *
import actionlib

from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_msgs.msg import Int32, Float32, Bool
from std_srvs.srv import Empty
from geometry_msgs.msg import Point
from nav_msgs.srv import GetMap
from ros_explore_navigation.msg import FollowPathAction, FollowPathGoal, FollowPathActionResult
from ros_explore_navigation.srv import GetPathLengths
from ros_explore_mapping.srv import GetDistance
from skeleton_disk_graph_roadmap.srv import GetDiskGraph

from sdg_roadmap_server import SDGRoadmapServer
from sdg_exploration_visualizer import SDGExplorationVisualizer


class SDGExplorationServer:
    def __init__(self, occ_map_service, pf_node_name, dist_server_node_name, map_frame, agent_frame, sdg_tuning_param, sdg_strat_params):
        # Subscribers
        self.pf_result_subscriber = rospy.Subscriber(
            pf_node_name + "/follow_path/result", FollowPathActionResult, self.pfResultCallback)
        # Publishers
        self.cancel_publisher = rospy.Publisher(
            pf_node_name + "/follow_path/cancel", GoalID, queue_size=1)
        self.search_dist_pub = rospy.Publisher("~frontiers_search_dist", Float32, queue_size=1)
        # Action server client
        self.follow_path_client = actionlib.SimpleActionClient(
            pf_node_name+"/follow_path", FollowPathAction)
        # Service proxy
        self.occ_map_proxy = rospy.ServiceProxy(occ_map_service, GetMap)
        self.dist_map_proxy = rospy.ServiceProxy(dist_server_node_name + "/get_distance", GetDistance)
        # Skeleton Disk-Graph planner as ROS server
        self.sdg_roadmap_server = SDGRoadmapServer(dist_server_node_name, map_frame, agent_frame, sdg_tuning_param)
        self.sdg_roadmap_strategy = SDGFrontiersStrategy(
            self.sdg_roadmap_server.sdg_planner,
            sdg_strat_params["min_frontier_unkn_ratio"],
            sdg_strat_params["max_frontier_unkn_ratio"],
            sdg_strat_params["unkn_occ_range"],
            sdg_strat_params["narrowness_cost_d_threshold"],
            sdg_strat_params["max_dist_compromise"],
            sdg_strat_params["unknown_max_coverage"],
            sdg_strat_params["frontiers_min_coverage"]
        )
        self.search_dist_increment = sdg_strat_params["search_dist_increment"]

        self.visualizer = SDGExplorationVisualizer()
        # State
        self.is_following_path = False
        self.current_target_id = None
        self.current_target_pos = None
        self.replan_pos = []
        self.past_plans = []

    
    def resetPlanner(self, sdg_tuning_param):
        rospy.logwarn("RESETTING PLANNER")
        self.sdg_roadmap_server.resetPlanner(sdg_tuning_param)



    def pfResultCallback(self, pfResultMsg):
        rospy.loginfo(str(pfResultMsg))
        self.active_target = None
        self.is_following_path = False

    def callPathFollowingDetached(self, goal_path):
        self.current_path = goal_path
        rospy.loginfo("Start path following")
        self.current_goal = self.follow_path_client.send_goal(goal_path)
        self.is_following_path = True

    def getValidFrontiersPaths(self):
        #SEARCH_DIST_INCREMENT = 10
        start = self.sdg_roadmap_server.agent_pos_listener.get2DAgentPos()
        self.replan_pos.append(start)
        self.visualizer.publishReplanPosViz(self.replan_pos)
        occ_map = EnvironmentMap.initFromRosEnvGridMapMsg(self.occ_map_proxy().map)
        dist_map = EnvironmentMap.initFromRosEnvGridMapMsg(self.dist_map_proxy().distance)
        frontiers_ids, frontiers_paths = self.sdg_roadmap_strategy.newGetValidFrontiers(start, occ_map, dist_map, self.search_dist_increment) #, self.min_frontier_unkn_ratio, unkn_range=self.unkn_occ_range, unkn_max_ratio=self.max_frontier_unkn_ratio)
        self.search_dist_pub.publish(self.sdg_roadmap_strategy.max_search_dist)
        return frontiers_ids, frontiers_paths

    def interruptOnInactiveTarget(self):
        occ_map = EnvironmentMap.initFromRosEnvGridMapMsg(self.occ_map_proxy().map)
        dist_map = EnvironmentMap.initFromRosEnvGridMapMsg(self.dist_map_proxy().distance)
        if self.current_target_pos is not None :
            target_rad = dist_map.valueAt(self.current_target_pos)
            if target_rad < self.sdg_roadmap_server.sdg_planner.tuning_parameters.min_pbubbles_rad :
                cancel_msg = GoalID()
                self.cancel_publisher.publish(cancel_msg)
                self.current_target_pos = None 
                self.current_target_id = None
                return
            
            # Interrrupt if target becomes known
            if not self.sdg_roadmap_strategy.checkFrontierValidity(self.current_target_pos, occ_map, dist_map):
                self.current_target_id = None
                rospy.loginfo("INTERRUPTING TARGET SEEN")
                cancel_msg = GoalID()
                self.cancel_publisher.publish(cancel_msg)
            

        '''
        if self.current_target_id is not None :
                if self.current_target_id in self.sdg_roadmap_server.sdg_planner.graph.nodes: 
                    #current_target_pos = self.sdg_roadmap_server.sdg_planner.graph.nodes[self.current_target_id]['node_obj'].pos
                    if not self.sdg_roadmap_strategy.checkFrontierValidity(self.current_target_pos, dist_map.valueAt(self.current_target_pos), occ_map, obst_thresh=95):
                        self.current_target_id = None
                        rospy.loginfo("INTERRUPTING TARGET SEEN")
                        cancel_msg = GoalID()
                        self.cancel_publisher.publish(cancel_msg)
        '''

    '''
    def selectPath(self):
        self.sdg_roadmap_server.updatePlanner(None)
        frontiers_ids, frontiers_paths = self.getValidFrontiersPaths()
        if not len(frontiers_paths):
            return None
        occ_map = EnvironmentMap.initFromRosEnvGridMapMsg(self.occ_map_proxy().map)
        #occ_map.display()
        #plt.show()
        path_costs = self.sdg_roadmap_strategy.getFrontiersPathsCosts(frontiers_ids, frontiers_paths, self.narrowness_cost_d_threshold, occ_map, unkn_range=self.unkn_occ_range)
        path_costs_coverage = {i: (1+2*path_costs[i]['coverage_reward'])/(1+10*path_costs[i]['energy_cost']+path_costs[i]['narrow_cost']) for i in path_costs}
        best_id = max(path_costs_coverage, key=path_costs_coverage.get)
        best_path = frontiers_paths[best_id]
        #best_path_id = min(path_costs[cid]['coverage_reward'] for cid in path_costs
        self.current_target_id = best_id
        return self.sdg_roadmap_server.constructPathMsg(best_path['postprocessed'])
    '''

    def newSelectPath(self):
        self.sdg_roadmap_server.updatePlanner(None)
        self.sdg_roadmap_strategy.updateRoadmap(self.sdg_roadmap_server.sdg_planner)
        time.sleep(0.25)
        frontiers_ids, frontiers_paths = self.getValidFrontiersPaths()
        if frontiers_paths is None:
            return None
        if not len(frontiers_paths):
            return None
        dvalid_frontiers_ids, dvalid_frontiers_paths = self.sdg_roadmap_strategy.filterFrontiersByTravelDist(frontiers_ids, frontiers_paths)
        occ_map = EnvironmentMap.initFromRosEnvGridMapMsg(self.occ_map_proxy().map)
        
        path_costs = self.sdg_roadmap_strategy.getFrontiersPathsCosts(dvalid_frontiers_ids, dvalid_frontiers_paths, occ_map)
        best_path = self.sdg_roadmap_strategy.selectPath(dvalid_frontiers_paths, path_costs)
        self.past_plans.append(best_path)
        #print(best_path)
        #path_dists_costs = [path_costs[p]['energy_cost'] for p in path_costs]
        #max_travel_dist = self.sdg_roadmap_strategy.max_dist_travel_compromise + np.min(path_dists_costs)
        #dist_valid_path_ids = [fp for fp in frontiers_paths if path_costs[fp]['energy_cost'] <= max_travel_dist]
        best_id = best_path['nodes_ids'][-1]
        self.current_target_id = best_id
        self.current_target_pos = self.sdg_roadmap_server.sdg_planner.graph.nodes[best_id]['node_obj'].pos
        #self.visualizer.publishFrontiersPlanViz(frontiers_paths, best_id, dist_valid_path_ids)
        self.visualizer.publishFrontiersPlanViz(frontiers_paths, best_id, dvalid_frontiers_ids)
        self.visualizer.publishPastPlanViz(self.past_plans[:-1])

        '''
        for ind in dvalid_frontiers_ids:
            if ind != best_id:
                self.sdg_roadmap_strategy.prev_frontiers_pos.append(self.sdg_roadmap_server.sdg_planner.graph.nodes[ind]['node_obj'].pos)
        '''
        return self.sdg_roadmap_server.constructPathMsg(best_path['postprocessed'])


if __name__ == "__main__":
    rospy.init_node("exploration_server")

    #slam_node_name = rospy.get_param("~slam_node_name", "rtabmap")
    occ_map_service = rospy.get_param("~occ_map_service", "/rtabmap/get_map")
    pf_node_name = rospy.get_param("~pf_node_name", "path_follower")
    
    dist_server_node_name = rospy.get_param("~dist_server_node_name","dist_server")
    map_frame = rospy.get_param("~map_frame","map")
    agent_frame = rospy.get_param("~agent_frame","base_link")

    sdg_tuning_param_dict = {
        "min_jbubbles_rad" : rospy.get_param("~sdg/min_jbubbles_rad",0.8),
        "min_pbubbles_rad" : rospy.get_param("~sdg/min_pbubbles_rad",0.4),
        "bubbles_dist_offset" : rospy.get_param("~sdg/bubbles_dist_offset",0.1),
        "knn_checks" : rospy.get_param("~sdg/knn_checks",40),
        "path_subdiv_length" : rospy.get_param("~sdg/path_subdiv_length",0.4)
    }
    sdg_tuning_param = SkeletonDiskGraphTuningParameters(sdg_tuning_param_dict)
    sdg_strategy_param_dict = {
        "min_frontier_unkn_ratio" : rospy.get_param("~strategy/min_frontier_unkn_ratio",0.2),
        "max_frontier_unkn_ratio" : rospy.get_param("~strategy/max_frontier_unkn_ratio",0.9),
        "unkn_occ_range" : rospy.get_param("~strategy/unkn_occ_range",[-1,-1]),
        "narrowness_cost_d_threshold" : rospy.get_param("~strategy/narrowness_cost_d_threshold",1.),
        "max_dist_compromise" : rospy.get_param("~strategy/max_dist_compromise",5),
        "unknown_max_coverage" : rospy.get_param("~strategy/unknown_max_coverage", 0.4),
        "frontiers_min_coverage" : rospy.get_param("~strategy/frontiers_min_coverage", 0),
        "search_dist_increment" : rospy.get_param("~strategy/search_dist_increment", 2)
    }

    exploration_server = SDGExplorationServer(occ_map_service, pf_node_name, dist_server_node_name, map_frame, agent_frame, sdg_tuning_param, sdg_strategy_param_dict)
    rospy.wait_for_service(dist_server_node_name + "/get_distance_skeleton")
    rospy.wait_for_service(dist_server_node_name + "/get_distance")
    exploration_server.follow_path_client.wait_for_server()

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        r.sleep()
        if not exploration_server.is_following_path:
            path = exploration_server.newSelectPath()
            if path is not None:
                exploration_server.callPathFollowingDetached(path)
            else:
                exploration_server.resetPlanner(sdg_tuning_param)
                #exit()
        else:
            exploration_server.interruptOnInactiveTarget()
    rospy.spin()