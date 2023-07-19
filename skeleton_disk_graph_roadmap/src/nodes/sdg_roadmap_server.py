#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import rospy
#from sdg_roadmap.sdg_roadmap_utils import *
from sdg_roadmap.skel_disk_graph_provider import *
from extended_mapping.map_processing import EnvironmentMap
from extended_mapping.ros_conversions import *
from navigation_utils.agent_pos_listener import AgentPosListener

from std_msgs.msg import Float64, Int32
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from extended_nav_mapping.srv import GetDistanceSkeleton
from skeleton_disk_graph_roadmap.msg import DiskGraph, DiskGraphNode, DiskGraphEdge
from skeleton_disk_graph_roadmap.srv import PlanPath, PlanPathResponse, GetDiskGraph, GetDiskGraphResponse

class SDGRoadmapServer:
    """
    ROS wrapper class to run the Skeleton Disk-Graph Roadmap Planner as a server node
    """

    def __init__(self, dist_server_node_name, map_frame, agent_frame, sdg_tuning_param):
        # TF listener
        self.agent_pos_listener = AgentPosListener(
            map_frame, agent_frame)
        # Publishers
        self.pathPublisher = rospy.Publisher("~planned_path", Path, queue_size=1, latch=True)
        self.graphPublisher = rospy.Publisher("~disk_graph", DiskGraph, queue_size=1, latch=True)
        self.graph_nodes_size_publisher = rospy.Publisher("~graph_size/nodes", Int32, queue_size=1, latch=True)
        self.graph_edges_size_publisher = rospy.Publisher("~graph_size/edges", Int32, queue_size=1, latch=True)
        # Services
        self.path_planning_service = rospy.Service("~plan_path", PlanPath, self.pathPlanningCallback)
        self.updateService = rospy.Service("~update_planner", Empty, self.updatePlanner)
        self.disk_graph_service = rospy.Service("~get_disk_graph", GetDiskGraph, self.getDiskGraphServiceCallback)
        # Services proxies 
        self.distance_serv_proxy = rospy.ServiceProxy(dist_server_node_name+"/get_distance_skeleton", GetDistanceSkeleton)

        # Initialize planner
        #self.sdg_planner = SkeletonDiskGraph(sdg_tuning_param)
        self.sdg_provider = SkeletonDiskGraphProvider(sdg_tuning_param_dict)
        self.path_subdiv_length = sdg_tuning_param_dict["path_subdiv_length"]

    
    # def resetPlanner(self, sdg_tuning_param_dict):
    #     self.sdg_planner = SkeletonDiskGraph(sdg_tuning_param_dict)
    

    def updatePlanner(self, req):
        """
        Queries new distance data and updates the planner accordingly
        """
        # Call distance server proxy
        dist_data = self.distance_serv_proxy()
        # Extract maps data
        dist_map = envMapFromRosEnvGridMapMsg(dist_data.distance)
        dist_skeleton = envMapFromRosEnvGridMapMsg(dist_data.skeleton)
        # Update planner accordingly
        self.sdg_provider.updateOnDistMapUpdate(dist_map, dist_skeleton, distance_change_update_thresh=0., res_tolerance=dist_map.resolution)
        self.publishDiskGraph()
        self.publishStats()
        rospy.loginfo("SDG planner updated")
        return EmptyResponse()

    def publishStats(self):
        """
        Publishes stats reflecting the planner state
        """
        nodes_size = len(self.sdg_provider.graph.nodes)
        edges_size = len(self.sdg_provider.graph.edges)
        self.graph_nodes_size_publisher.publish(Int32(nodes_size))
        self.graph_edges_size_publisher.publish(Int32(edges_size))


    # def outdated_pathPlanningCallback(self, path_planning_request):
    #     """
    #     Computes and returns a feasible path on a new path planning request
    #     """
    #     self.updatePlanner(None)
    #     start = self.agent_pos_listener.get2DAgentPos()
    #     goal = np.array([path_planning_request.goal.x,
    #                     path_planning_request.goal.y])
    #     world_path = self.sdg_planner.getWorldPath(start, goal)
    #     path, path_radii = world_path['postprocessed']['pos'], world_path['postprocessed']['radii']
    #     reduced_path = self.sdg_planner.reduceBubblesPath(path, path_radii)
    #     rospy.loginfo("Path planning - goal : [{},{}], found path : {}".format(
    #         goal[0], goal[1], (path is not None)))
    #     print("Got path - starting reduction")
    #     if path is None:
    #         return PlanPathResponse()
    #     path = np.array(path)
    #     #red_path = path
    #     red_path = np.array(self.sdg_planner.reduceBubblesPath(path, path_radii))
    #     red_path_radii = np.array([self.sdg_planner.cached_dist_map.valueAt(wp) for wp in red_path])
    #     path_cpoints = computeSplineControlPoints(red_path, red_path_radii)
    #     spline_path = computeSplinePath(path_cpoints)
    #     red_path = spline_path.waypoints
    #     print("Got reduced path")
    #     return self.constructPathMsg(red_path)

    def pathPlanningCallback(self, path_planning_request):
        """
        Computes and returns a feasible path on a new path planning request
        """
        self.updatePlanner(None)
        start = self.agent_pos_listener.get2DAgentPos()
        goal = np.array([path_planning_request.goal.x,
                        path_planning_request.goal.y])
        world_path = self.sdg_provider.getWorldPath(start, goal)
        if world_path is None:
            rospy.logwarn("Planning failed")
            return PlanPathResponse()
        simplified_path = world_path.getSubdivized(int(world_path.getTotalLength()/self.path_subdiv_length), self.sdg_provider.cached_dist_map).getReducedBubbles()
        spline_path = simplified_path.getSmoothedSpline(self.sdg_provider.cached_dist_map)
        rospy.loginfo("Planning succesful")
        return self.constructPathMsg(spline_path.waypoints)

    def getDiskGraphServiceCallback(self, req):
        response = GetDiskGraphResponse()
        response.disk_graph = self.constructGraphMsg()
        return response
    
    def constructPathMsg(self, waypoints):
        path_msg = Path()
        path_header = Header()
        path_header.frame_id = "map"
        path_header.stamp = rospy.Time(0)
        path_msg.header = path_header
        path_poses = []
        for k, path_pos in enumerate(waypoints):
            pos_header = Header()
            pos_header.frame_id = "map"
            pos_header.stamp = rospy.Time(0)
            pos_stamped = PoseStamped()
            pos_stamped.header = pos_header
            pos_stamped.pose = Pose()
            pos_stamped.pose.position.x = path_pos[0]
            pos_stamped.pose.position.y = path_pos[1]
            pos_stamped.pose.orientation.w = 1
            path_poses.append(pos_stamped)
        path_msg.poses = path_poses
        self.pathPublisher.publish(path_msg)
        # planner_rrt.show()

        response_path = PlanPathResponse()
        response_path.path = path_msg

        return response_path


    def constructGraphMsg(self):
        """Builds and returns a skeleton_disk_graph_roadmap.msg/DiskGraph message representing the current graph state"""
        if not len(self.sdg_provider.graph.nodes):
            return
        else:
            graph_msg = DiskGraph()
            graph_msg.nodesIdsMapping = self.sdg_provider.nodes_ids
            graph_msg_nodes = []
            nodes = [self.sdg_provider.graph.nodes[nid]["node_obj"]
                     for nid in self.sdg_provider.nodes_ids]
            #adj_matrix = np.zeros([len(nodes), len(nodes)], int)
            for node in nodes:
                graph_node = DiskGraphNode(
                    node.id, node.pos[0], node.pos[1], node.bubble_rad)
                graph_msg_nodes.append(graph_node)

            graph_msg_edges = []
            for e in self.sdg_provider.graph.edges:
                edge = DiskGraphEdge(e[0], e[1])
                graph_msg_edges.append(edge)
            graph_msg.edges = list(graph_msg_edges)
            graph_msg.nodes = graph_msg_nodes
            return graph_msg

    def publishDiskGraph(self):
        """
        Publishes the planner current disk graph as a ROS message 
        """
        graph_msg = self.constructGraphMsg()
        if graph_msg is not None:
            self.graphPublisher.publish(graph_msg)


# Main
if __name__ == '__main__':
    rospy.init_node("sdg_roadmap")
    r = rospy.Rate(1)
    
    dist_server_node_name = rospy.get_param("~dist_server_node_name","dist_server")
    map_frame = rospy.get_param("~map_frame","map")
    #agent_frame = "corrected_base_link"
    agent_frame = rospy.get_param("~agent_frame","base_link")

    sdg_tuning_param_dict = {
        "min_jbubbles_rad" : rospy.get_param("~sdg/min_jbubbles_rad",0.2),
        "min_pbubbles_rad" : rospy.get_param("~sdg/min_pbubbles_rad",0.15),
        "bubbles_dist_offset" : rospy.get_param("~sdg/bubbles_dist_offset",0.1),
        "knn_checks" : rospy.get_param("~sdg/knn_checks",40),
        "path_subdiv_length" : rospy.get_param("~sdg/path_subdiv_length",0.4)
    }
    #sdg_tuning_param = SkeletonDiskGraphTuningParameters(sdg_tuning_param_dict)
    sdg_roadmap = SDGRoadmapServer(dist_server_node_name, map_frame, agent_frame, sdg_tuning_param_dict)
    rospy.wait_for_service(dist_server_node_name + "/get_distance_skeleton")

    while not rospy.is_shutdown():
        r.sleep()
        # DO STUFF HERE
        #print("Skeleton Disk Graph active")
    rospy.spin()
    