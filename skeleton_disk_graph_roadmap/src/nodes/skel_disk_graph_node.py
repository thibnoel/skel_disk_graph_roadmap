#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import rospy
import actionlib
from sdg_roadmap.skel_disk_graph_provider import *
from sdg_roadmap.ros_conversions import *
from extended_mapping.map_processing import EnvironmentMap
from extended_mapping.ros_conversions import *
from navigation_utils.agent_pos_listener import AgentPosListener
from navigation_utils.ros_conversions import *

from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_msgs.msg import Float64, Int32
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetMap
from extended_navigation_mapping.msg import FollowPathAction, FollowPathGoal, FollowPathActionResult
from extended_navigation_mapping.srv import GetDistance, GetDistanceSkeleton
from skeleton_disk_graph_roadmap.msg import DiskGraph, DiskGraphNode, DiskGraphEdge
from skeleton_disk_graph_roadmap.srv import PlanPath, PlanPathResponse, GetDiskGraph, GetDiskGraphResponse, NavigateTo, NavigateToResponse

from sdg_ros_components import *
from sdg_ros_visualization import *

class SkelDiskGraphServer:
    def __init__(self, distance_mapping_server, map_frame, agent_frame, sdg_parameters, visualizer=None):
        # TF listener
        self.agent_pos_listener = AgentPosListener(map_frame, agent_frame)
        # Publishers
        self.graphPublisher = rospy.Publisher("~disk_graph", DiskGraph, queue_size=1, latch=True)
        self.graph_nodes_size_publisher = rospy.Publisher("~logging/graph_size/nodes", Int32, queue_size=1, latch=True)
        self.graph_edges_size_publisher = rospy.Publisher("~logging/graph_size/edges", Int32, queue_size=1, latch=True)
        # Services
        self.updateService = rospy.Service("~update_planner", Empty, self.updatePlanner)
        self.disk_graph_service = rospy.Service("~get_disk_graph", GetDiskGraph, self.getDiskGraphServiceCallback)
        
        # Services proxies
        self.distance_serv_proxy = rospy.ServiceProxy(distance_mapping_server+"/get_distance_skeleton", GetDistanceSkeleton)

        # Initialize planner
        self.sdg_provider = SkeletonDiskGraphProvider(sdg_parameters)
        self.visualizer = visualizer

    def updatePlanner(self, req):
        """
        Queries new distance data and updates the planner accordingly
        """
        dim_check = False
        while not dim_check:
            # Call distance server proxy
            dist_data = self.distance_serv_proxy()
            # Extract maps data
            dist_map = envMapFromRosEnvGridMapMsg(dist_data.distance)
            dist_skeleton = envMapFromRosEnvGridMapMsg(dist_data.skeleton)
            dim_check = (dist_map.dim == dist_skeleton.dim)
        
        # Update planner accordingly
        self.sdg_provider.updateOnDistMapUpdate(dist_map, dist_skeleton, distance_change_update_thresh=0., res_tolerance=dist_map.resolution)
        rospy.loginfo("SDG planner updated")
        self.publishDiskGraph()
        self.publishGraphSize()
        if self.visualizer is not None:
            self.visualizer.publishNodesVizMsg(self.sdg_provider)
            self.visualizer.publishEdgesVizMsg(self.sdg_provider)
        return EmptyResponse()

    def publishGraphSize(self):
        """
        Publishes the graph size
        """
        nodes_size = len(self.sdg_provider.graph.nodes)
        edges_size = len(self.sdg_provider.graph.edges)
        self.graph_nodes_size_publisher.publish(Int32(nodes_size))
        self.graph_edges_size_publisher.publish(Int32(edges_size))

    def getDiskGraphServiceCallback(self, req):
        response = GetDiskGraphResponse()
        response.disk_graph = skelDiskGraphToDiskGraphMsg(self.sdg_provider)
        return response

    def publishDiskGraph(self):
        """
        Publishes the planner current disk graph as a ROS message 
        """
        graph_msg = skelDiskGraphToDiskGraphMsg(self.sdg_provider)
        if graph_msg is not None:
            self.graphPublisher.publish(graph_msg)

# Main
if __name__ == '__main__':
    rospy.init_node("sdg_server")
    r = rospy.Rate(1)
    
    distance_mapping_server = rospy.get_param("~distance_mapping_server","dist_server")
    map_frame = rospy.get_param("~map_frame","map")
    agent_frame = rospy.get_param("~agent_frame","base_link")

    # SDG construction parameters
    sdg_base_parameters = {
        "min_jbubbles_rad" : rospy.get_param("~sdg/min_jbubbles_rad",0.2),
        "min_pbubbles_rad" : rospy.get_param("~sdg/min_pbubbles_rad",0.15),
        "bubbles_dist_offset" : rospy.get_param("~sdg/bubbles_dist_offset",0.1),
        "knn_checks" : rospy.get_param("~sdg/knn_checks",40),
    }
    # Navigation parameters
    sdg_nav_parameters = {
        "active": rospy.get_param("~sdg/navigation/active", True),
        "path_follower_server": rospy.get_param("~sdg/navigation/path_follower_server", "path_follower"),
        "path_safety_distance": rospy.get_param("~sdg/navigation/path_safety_distance", 0.1)
    }
    # Exploration parameters
    sdg_explo_parameters = {
        "active": rospy.get_param("~sdg/exploration/active", True),
        "occupancy_map_service": rospy.get_param("~occupancy_map_service", "/rtabmap/get_map"),
        "strategy": {
            "unkn_occ_range" : rospy.get_param("~sdg/exploration/strategy/unkn_occ_range",[-1,-1]),
            "frontiers_max_known_ratio" : rospy.get_param("~sdg/exploration/strategy/frontiers_max_known_ratio", 0.5),
            "search_dist_increment" : rospy.get_param("~sdg/exploration/strategy/search_dist_increment", 2),
            "path_cost_parameters": rospy.get_param("~sdg/exploration/strategy/path_cost_parameters", "")
        }
    }
    start_exploration_paused = rospy.get_param("~sdg/exploration/start_paused", True)
    # Visualization parameters
    sdg_viz_parameters = {
        "active": rospy.get_param("~sdg/visualization/active", True),
        "roadmap": {
            "graph_edges_color": rospy.get_param("~sdg/visualization/roadmap/edges/color", (1,0,0,1)),
            "graph_edges_linewidth": rospy.get_param("~sdg/visualization/roadmap/edges/linewidth", 0.1),
            "graph_nodes_color": rospy.get_param("~sdg/visualization/roadmap/nodes/color", (1,0,0,1)),
            "graph_nodes_linewidth": rospy.get_param("~sdg/visualization/roadmap/nodes/linewidth", 0.1),
        },
        "navigation": {
            "path_nodes_color": rospy.get_param("~sdg/visualization/navigation/path_nodes/color", (1,0,0,1)),
            "path_nodes_linewidth_path": rospy.get_param("~sdg/visualization/navigation/path_nodes/path_linewidth", 0.1),
            "path_nodes_linewidth_nodes": rospy.get_param("~sdg/visualization/navigation/path_nodes/nodes_linewidth", 0.05),
        },
        "exploration": {
            "path_selection_linewidth": rospy.get_param("~sdg/visualization/exploration/path_selection/linewidth", 0.1),
            "path_selection_cmap": rospy.get_param("~sdg/visualization/exploration/path_selection/cmap", "viridis"),
            "path_selection_labels_size": rospy.get_param("~sdg/visualization/exploration/path_selection/labels_size", 0.5),
            "curr_target_linewidth": rospy.get_param("~sdg/visualization/exploration/curr_target/linewidth", 0.2),
            "curr_target_color": rospy.get_param("~sdg/visualization/exploration/curr_target/color", (1,0,0,1)),
        }
    }
    viz_component = None
    if sdg_viz_parameters["active"]:
        viz_component = SDGVisualizationComponent(sdg_viz_parameters)

    sdg_server = SkelDiskGraphServer(distance_mapping_server, map_frame, agent_frame, sdg_base_parameters, visualizer=viz_component)
    rospy.wait_for_service(distance_mapping_server + "/get_distance_skeleton")

    planning_component = SDGPlanningComponent(sdg_server, visualizer=viz_component)
    if sdg_nav_parameters["active"]:
        nav_component = SDGNavigationComponent(sdg_server, planning_component, sdg_nav_parameters["path_follower_server"], sdg_nav_parameters["path_safety_distance"])
    if sdg_explo_parameters["active"]:
        explo_component = SDGExplorationComponent(sdg_server, planning_component, nav_component, sdg_explo_parameters["strategy"], sdg_explo_parameters["occupancy_map_service"], start_paused=start_exploration_paused, visualizer=viz_component)   

    while not rospy.is_shutdown():
        r.sleep()
        if sdg_nav_parameters["active"]:
            nav_component.update()
        if sdg_explo_parameters["active"]:
            explo_component.update()    
    rospy.spin()