#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import rospy
import numpy as np
from extended_mapping.map_processing import EnvironmentMap
from nav_utilities import agent_pos_listener

from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path
from extended_navigation_mapping.srv import GetDistance
from skeleton_disk_graph_roadmap.msg import DiskGraph, DiskGraphNode, DiskGraphEdge
from skeleton_disk_graph_roadmap.srv import PlanPath, PlanPathResponse
from visualization_msgs.msg import Marker, MarkerArray

class RVizCirclesProvider:
    def __init__(self, n_interp=200):
        theta = np.arange(0,2*np.pi,2*np.pi/n_interp)
        self.interp_points = np.array([np.cos(theta), np.sin(theta)]).T

    def getCircMarker(self, marker_id, center, radius, height, color, line_width, lifetime=0):
        points = [] # linked as 0-1, 2-3 etc
        for k, p in enumerate(self.interp_points[:-1]):
            points.append(Point(radius*p[0], radius*p[1], 0))
            points.append(Point(radius*self.interp_points[k+1][0], radius*self.interp_points[k+1][1], 0))
        points.append(Point(radius*self.interp_points[-1][0], radius*self.interp_points[-1][1], 0))
        points.append(Point(radius*self.interp_points[0][0], radius*self.interp_points[0][1], 0))
        
        #markers_list = []
        #marker_ind = 0
        #for k,p in enumerate(self.interp_points[:-1]):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = marker_id
        marker.type = 5 # Line list
        marker.pose = Pose(Point(center[0],center[1],height), Quaternion(0,0,0,1))
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.scale.x = line_width
        marker.points = points
        marker.lifetime = rospy.Duration(lifetime)
        #markers_list.append(marker)
        #marker_ind += 1
        #return MarkerArray(markers_list)
        return marker

class SDGRoadmapVisualizer:
    """
    ROS wrapper class to visualize the Skeleton Disk-Graph Roadmap Planner in RViz
    """

    def __init__(self, sgd_node_name):
        # Subscribers
        self.graph_subscriber = rospy.Subscriber(sgd_node_name+"/disk_graph", DiskGraph, self.graphCallback, queue_size=1)

        # Publishers
        self.nodesVizPublisher = rospy.Publisher("~nodes_viz", MarkerArray, queue_size=1, latch=True)
        self.edgesVizPublisher = rospy.Publisher("~edges_viz", MarkerArray, queue_size=1, latch=True)

        self.circles_provider = RVizCirclesProvider()

        # Parameters
        #self.bubbles_color = [0.4235, 0.651, 0.7569, 1]
        self.bubbles_color = [0.25, 0.55, 0.9, 0.5]
        self.edges_color = [0.25, 0.55, 0.9, 1]
        self.edges_width = 0.05
    
    
    def graphCallback(self, graph_msg):
        #if self.segmentation_display_last_date is not None:
        #    if rospy.Time.now() - self.segmentation_display_last_date < rospy.Duration(self.SEG_DISPLAY_DURATION):
        #        return
        """Graph topic subscriber callback - updates the state of the current graph"""
        #zeroCell_pos = np.array([
        #    self.current_map_data.origin.position.x,
        #    self.current_map_data.origin.position.y
        #])
        nodes_ids_mapping = np.array(graph_msg.nodesIdsMapping)
        nodes_pos = np.array([[node.pos_x, node.pos_y]
                             for node in graph_msg.nodes])
        #self.graph_nodes_pos = np.array([pixPos_to_worldPos(
        #    p, zeroCell_pos, self.current_map_data.resolution) for p in nodes_pos])
        nodes_rad = [node.bubble_rad for node in graph_msg.nodes]
        #self.graph_nodes_rad = np.array(
        #    nodes_rad)*self.current_map_data.resolution
        graph_edges = [[e.n0_id, e.n1_id] for e in graph_msg.edges]

        B_SCALE = 3*[0.05]
        self.nodesVizPublisher.publish(self.constructClearMsg())
        self.edgesVizPublisher.publish(self.constructClearMsg())
        self.nodesVizPublisher.publish(
            self.constructNodesVizMsg(nodes_pos, nodes_rad, self.bubbles_color, B_SCALE, height_offset=-B_SCALE[2]+0.2))
        self.edgesVizPublisher.publish(
            self.constructEdgesVizMsg(nodes_pos, graph_edges, nodes_ids_mapping,
                                      self.edges_color, offset_height=0.2, width=self.edges_width))

    def constructClearMsg(self):
        """Builds an empty RViz visualization message which clears all markers"""
        marker = Marker()
        marker.action = 3
        return [marker]

    def constructNodesVizMsg(self, bubbles_pos, bubbles_rad, color, scale, height_offset=0, id_offset=0):
        """Builds a RViz visualization message for the graph nodes (bubbles)"""
        if not len(bubbles_pos):
            return

        #bubbles_rad = self.graph_nodes_rad
        #bubbles_pos = self.graph_nodes_pos
        
        markers_list = []
        '''
        for k, b in enumerate(bubbles_pos):
            bcolor = color
            bhoffset = height_offset

            vertex_marker = Marker()
            marker_header = Header()
            marker_header.frame_id = "map"
            marker_header.stamp = rospy.Time(0)
            vertex_marker.header = marker_header
            vertex_marker.id = k + id_offset
            vertex_marker.type = 3  # cylinder
            vertex_marker.pose = Pose(
                Point(b[0], b[1], bhoffset), Quaternion(0, 0, 0, 1))
            #color = colors(1.*v.level/max_vertex_level)

            vertex_marker.color.r = bcolor[0]
            vertex_marker.color.g = bcolor[1]
            vertex_marker.color.b = bcolor[2]
            vertex_marker.color.a = bcolor[3]
            vertex_marker.scale.x = 2*bubbles_rad[k] - 0.05
            vertex_marker.scale.y = 2*bubbles_rad[k] - 0.05
            vertex_marker.scale.z = scale[2]

            vmarker2 = Marker()
            vmarker2.header = marker_header
            vmarker2.id = k+10000 + id_offset
            vmarker2.type = 3
            vmarker2.pose = Pose(Point(b[0], b[1], bhoffset), Quaternion(0, 0, 0, 1))
            vmarker2.color.r = 52./255
            vmarker2.color.g = 52./255
            vmarker2.color.b = 52./255
            vmarker2.color.a = 1
            vmarker2.scale.x = 2*bubbles_rad[k]
            vmarker2.scale.y = 2*bubbles_rad[k]
            vmarker2.scale.z = 0.5*scale[2]

            #vertex_marker.points = [Point(v.pos[0], v.pos[1], 0), Point(v.parent.pos[0], v.parent.pos[1], 0)]
            vertex_marker.lifetime = rospy.Duration(0)
            vmarker2.lifetime = rospy.Duration(0)
            markers_list.append(vertex_marker)
            markers_list.append(vmarker2)
        marker_array = MarkerArray(markers_list)
        #color[3] = 1
        return marker_array
        '''
        for k, pos in enumerate(bubbles_pos) :
            circle = self.circles_provider.getCircMarker(k, pos, bubbles_rad[k], height_offset, color, scale[2], lifetime=0)
            markers_list.append(circle)
        return MarkerArray(markers_list)
        

    def constructEdgesVizMsg(self, bubbles_pos, graph_edges, nodes_ids_mapping, color, offset_height=.1, width=.06, id_offset=0):
        """Builds a RViz visualization message for the graph edges"""
        #nodes_list = self.graph_nodes_pos
        #edges_list = self.graph_edges

        markers_list = []
        marker_ind = 0
        # print(nodes_list)
        # print(edges_list)
        for e in graph_edges:
            #ni = nodes_list[self.nodes_ids_mapping[e[0]]]
            ni = bubbles_pos[np.argwhere(nodes_ids_mapping == e[0])[0]][0]
            nj = bubbles_pos[np.argwhere(nodes_ids_mapping == e[1])[0]][0]
            line = np.array([ni, nj])
            print(ni, nj, line)

            vertex_marker = Marker()
            marker_header = Header()
            marker_header.frame_id = "map"
            marker_header.stamp = rospy.Time(0)
            vertex_marker.header = marker_header
            vertex_marker.id = marker_ind + id_offset
            vertex_marker.type = 4  # line strip
            vertex_marker.pose = Pose(
                Point(0, 0, offset_height), Quaternion(0, 0, 0, 1))
            #color = [0.4,0.6,1,1]
            vertex_marker.color.r = color[0]
            vertex_marker.color.g = color[1]
            vertex_marker.color.b = color[2]
            vertex_marker.color.a = color[3]
            vertex_marker.scale.x = width
            vertex_marker.points = [
                Point(line[0][0], line[0][1], 0), Point(line[1][0], line[1][1], 0)]
            vertex_marker.lifetime = rospy.Duration(0)
            markers_list.append(vertex_marker)
            #plt.plot(line[:,0], line[:,1], color=graph_color, lw=lw)
            marker_ind += 1

        marker_array = MarkerArray(markers_list)
        return marker_array


# Main
if __name__ == '__main__':
    rospy.init_node("sdg_viz")
    r = rospy.Rate(1)

    sdg_node_name = "sdg_roadmap"
    #sdg_node_name = "exploration_server"
    sdg_visualizer = SDGRoadmapVisualizer(sdg_node_name)

    while not rospy.is_shutdown():
        r.sleep()
    rospy.spin()