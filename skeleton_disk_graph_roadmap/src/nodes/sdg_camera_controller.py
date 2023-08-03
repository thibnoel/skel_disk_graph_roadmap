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
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import Point, Vector3
from nav_msgs.srv import GetMap
from extended_navigation_mapping.msg import FollowPathAction, FollowPathGoal, FollowPathActionResult
from extended_navigation_mapping.srv import GetDistance
from skeleton_disk_graph_roadmap.srv import GetDiskGraph

from sdg_roadmap_server import SDGRoadmapServer
from sdg_exploration_visualizer import SDGExplorationVisualizer

class SDGNeighborsCamTargetSelector:
    def __init__(self, sdg_provider, agent_pos_listener, cam_target_topic):
        self.sdg_provider = sdg_provider
        self.agent_pos_listener = agent_pos_listener
        # Target publisher
        self.target_publisher = rospy.Publisher(cam_target_topic, Vector3, queue_size=1)
        # Parameters
        self.min_look_duration = 3
        # State
        self.current_target_id = None
        self.last_update_t = None

    def getNeighbors(self):
        curr_pos = self.agent_pos_listener.get2DAgentPos()
        if not len(self.sdg_provider.graph.nodes):
            return None
        curr_ref_bubble = self.sdg_provider.biggestContainingBubble(curr_pos)
        neighb = []
        if curr_ref_bubble is not None:
            #neighb = [self.sdg_provider.graph.nodes[i]["node_obj"] for i in list(self.sdg_provider.graph.neighbors(curr_ref_bubble.id))]
            neighb = list(self.sdg_provider.graph.neighbors(curr_ref_bubble.id))
        return neighb

    def evaluateNeighbors(self, neighbors_ids, occupancy_map, map_unkn_range):
        eval_dict = {}
        for n_id in neighbors_ids:
            neighb_bubble = self.sdg_provider.graph.nodes[n_id]['node_obj']
            neighb_coverage = neighb_bubble.computeBubbleCoverage(occupancy_map, unkn_range=map_unkn_range, inflation_rad_mult=0.9)
            eval_dict[n_id] = neighb_coverage*neighb_bubble.bubble_rad*neighb_bubble.bubble_rad
        return eval_dict

    def updateTarget(self, occupancy_map, map_unkn_range):
        # if self.current_target_id is not None:
        #     target_bubble = self.sdg_provider.graph.nodes[self.current_target_id]['node_obj']
        #     coverage = target_bubble.computeBubbleCoverage(occupancy_map, unkn_range=map_unkn_range, inflation_rad_mult=0.9)
        #     if coverage > 0.8:
        #         self.current_target_id = None
        # else:
        neighb_ids = self.getNeighbors()
        if neighb_ids is None:
            return
        eval_dict = self.evaluateNeighbors(neighb_ids, occupancy_map, map_unkn_range)
        t = rospy.Time.now()
        dt = 0
        if self.last_update_t is not None:
            dt  = (t - self.last_update_t).to_sec()
        if len(eval_dict) and (self.last_update_t is None or dt > self.min_look_duration):
            best_neighbor = min(eval_dict, key=eval_dict.get)
            self.current_target_id = best_neighbor
            self.publishTarget()
            self.last_update_t = rospy.Time.now()

    def publishTarget(self):
        if self.current_target_id in self.sdg_provider.graph.nodes:
            target_bubble = self.sdg_provider.graph.nodes[self.current_target_id]['node_obj']
            target = Vector3(target_bubble.pos[0], target_bubble.pos[1], 0.2)
            self.target_publisher.publish(target)