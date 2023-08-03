import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from extended_mapping.map_processing import EnvironmentMap
from navigation_utils.ros_conversions import *
from navigation_utils import agent_pos_listener

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
        """
        Returns a circle visualization_msg/Marker 
        """
        points = [] # linked as 0-1, 2-3 etc
        # Create Points
        for k, p in enumerate(self.interp_points[:-1]):
            points.append(Point(radius*p[0], radius*p[1], 0))
            points.append(Point(radius*self.interp_points[k+1][0], radius*self.interp_points[k+1][1], 0))
        points.append(Point(radius*self.interp_points[-1][0], radius*self.interp_points[-1][1], 0))
        points.append(Point(radius*self.interp_points[0][0], radius*self.interp_points[0][1], 0))
        # Create RViz marker
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
        return marker

def getLineListMarker(segments, color, line_width, height, lifetime=0):
    """
    Returns a visualization_msg/Marker for a list of segments 
    """
    points = [] # linked as 0-1, 2-3 etc
    # Create Points
    for k, seg in enumerate(segments):
        points.append(Point(seg[0][0], seg[0][1], 0))
        points.append(Point(seg[1][0], seg[1][1], 0))
    # Create RViz marker
    marker = Marker()
    marker.header = Header()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.id = 0
    marker.type = 5 # Line list
    marker.pose = Pose(Point(0,0,height), Quaternion(0,0,0,1))
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.scale.x = line_width
    marker.points = points
    marker.lifetime = rospy.Duration(lifetime)
    return marker

def getBubblesPathMarkerArray(bubbles_path, color, path_line_width, nodes_line_width, height, show_nodes=True, nodes_subsamp_ratio=1, lifetime=0, id_offset=0, circles_provider=None):
    points = [] # linked as 0-1, 1-2 etc
    circles = []
    # Create Points
    for k, wp in enumerate(bubbles_path.waypoints):
        points.append(Point(wp[0], wp[1], 0))
        rate = int(1/nodes_subsamp_ratio)
        if show_nodes and (circles_provider is not None) and (k%rate == 0):
            circ = circles_provider.getCircMarker(k+id_offset+1, wp, bubbles_path.radii[k], 0, color, nodes_line_width, lifetime=lifetime)
            circles.append(circ)
    # Create RViz marker
    marker = Marker()
    marker.header = Header()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.id = id_offset
    marker.type = 4 # Line strip
    marker.pose = Pose(Point(0,0,height), Quaternion(0,0,0,1))
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.scale.x = path_line_width
    marker.points = points
    marker.lifetime = rospy.Duration(lifetime)
    return MarkerArray([marker, *circles])

def getTextMarker(marker_id, ref_pos, height, text, color, size=0.5, pos_offset=np.array([0.2,0.2]), lifetime=0):
    text_marker = Marker()
    text_marker.header = Header()
    text_marker.header.frame_id = "map"
    text_marker.header.stamp = rospy.Time(0)
    text_marker.type = 9  # text
    text_marker.color.r = color[0]
    text_marker.color.g = color[1]
    text_marker.color.b = color[2]
    text_marker.color.a = color[3]
    text_marker.scale.z = size
    text_marker.pose = Pose(Point(ref_pos[0], ref_pos[1], height), Quaternion(0, 0, 0, 1))
    text_marker.pose.position.x += pos_offset[0]
    text_marker.pose.position.y += pos_offset[1]
    text_marker.id = marker_id
    text_marker.text = text
    text_marker.lifetime = rospy.Duration(lifetime)
    return text_marker



class SDGVisualizationComponent:
    def __init__(self, visualization_parameters):
        self.circles_provider = RVizCirclesProvider()
        ### Publishers
        # Roadmap graph
        self.nodes_viz_publisher = rospy.Publisher("~viz/roadmap_nodes", MarkerArray, queue_size=1, latch=True)
        self.edges_viz_publisher = rospy.Publisher("~viz/roadmap_edges", MarkerArray, queue_size=1, latch=True)
        # Planning
        self.path_nodes_viz_publisher = rospy.Publisher("~viz/nav_path_nodes", MarkerArray, queue_size=1, latch=True)
        # Exploration
        self.path_selection_viz_publisher = rospy.Publisher("~viz/exploration_path_selection", MarkerArray, queue_size=1, latch=True)
        self.current_target_viz_publisher = rospy.Publisher("~viz/exploration_current_target", MarkerArray, queue_size=1, latch=True)
        # Visualization parameters
        self.graph_edges_color = visualization_parameters["roadmap"]["graph_edges_color"]
        self.graph_edges_linewidth = visualization_parameters["roadmap"]["graph_edges_linewidth"]
        self.graph_nodes_color = visualization_parameters["roadmap"]["graph_nodes_color"]
        self.graph_nodes_linewidth = visualization_parameters["roadmap"]["graph_nodes_linewidth"]
        self.path_nodes_color = visualization_parameters["navigation"]["path_nodes_color"]
        self.path_nodes_linewidth_path = visualization_parameters["navigation"]["path_nodes_linewidth_path"]
        self.path_nodes_linewidth_nodes = visualization_parameters["navigation"]["path_nodes_linewidth_nodes"]
        self.exploration_path_selection_linewidth = visualization_parameters["exploration"]["path_selection_linewidth"]
        self.exploration_path_selection_cmap = visualization_parameters["exploration"]["path_selection_cmap"]
        exec("self.exploration_path_selection_cmap = plt.cm.{}".format(self.exploration_path_selection_cmap))
        self.exploration_path_selection_labels_size = visualization_parameters["exploration"]["path_selection_labels_size"]
        self.exploration_curr_target_linewidth = visualization_parameters["exploration"]["curr_target_linewidth"]
        self.exploration_curr_target_color = visualization_parameters["exploration"]["curr_target_color"]
        self.clearAll()

    def constructClearMsg(self, lifetime=0.1):
        """Builds an empty RViz visualization message which clears all markers"""
        marker = Marker()
        marker.action = 3
        marker.lifetime = rospy.Duration(lifetime)
        return [marker]

    def clearAll(self):
        self.nodes_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.edges_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.path_nodes_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.path_selection_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.current_target_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        
    def publishNodesVizMsg(self, sdg_provider, height_offset=0):
        """Builds a RViz visualization message for the graph nodes (bubbles)"""
        if not len(sdg_provider.graph.nodes):
            return
        bubbles = [sdg_provider.graph.nodes[i]['node_obj'] for i in sdg_provider.graph.nodes]
        bubbles_pos = [b.pos for b in bubbles]
        bubbles_rad = [b.bubble_rad for b in bubbles]
        markers_list = []
        for k, pos in enumerate(bubbles_pos) :
            circle = self.circles_provider.getCircMarker(k, pos, bubbles_rad[k], height_offset, self.graph_nodes_color, self.graph_nodes_linewidth, lifetime=0)
            markers_list.append(circle)
        self.nodes_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.nodes_viz_publisher.publish(MarkerArray(markers_list))

    def publishEdgesVizMsg(self, sdg_provider, height_offset=0):
        """Builds a RViz visualization message for the graph nodes (bubbles)"""
        if not len(sdg_provider.graph.edges):
            return
        segments = []
        for e in sdg_provider.graph.edges:
            start = sdg_provider.graph.nodes[e[0]]['node_obj'].pos
            end = sdg_provider.graph.nodes[e[1]]['node_obj'].pos
            segments.append([start, end])
        markers_list = [getLineListMarker(segments, self.graph_edges_color, self.graph_edges_linewidth, height_offset, lifetime=0)]
        self.edges_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.edges_viz_publisher.publish(MarkerArray(markers_list))

    def publishPlanPathVizMsg(self, bubbles_path, show_nodes=True, height_offset=0, nodes_subsamp_ratio=0.2):
        marker_array = getBubblesPathMarkerArray(bubbles_path, self.path_nodes_color, self.path_nodes_linewidth_path, self.path_nodes_linewidth_nodes, height_offset, show_nodes=show_nodes, nodes_subsamp_ratio=nodes_subsamp_ratio,  lifetime=0, id_offset=0, circles_provider=self.circles_provider)
        id_offset = marker_array.markers[-1].id + 1
        outline_markers_array = getBubblesPathMarkerArray(bubbles_path, (0,0,0,1), self.path_nodes_linewidth_path*1.8, 0, height_offset -0.01, show_nodes=False, lifetime=0, id_offset=id_offset)
        marker_array.markers.extend(outline_markers_array.markers)
        self.path_nodes_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.path_nodes_viz_publisher.publish(marker_array)

    def publishPathSelectionViz(self, frontiers_paths, height_offset=0, lifetime=4, outline=False):
        markers_list = []
        #marker_index = 0
        id_offset = 0
        path_costs = [frontiers_paths[f_id]['aggregated_cost'] for f_id in frontiers_paths]
        norm = Normalize(vmin=np.min(path_costs), vmax=np.max(path_costs))

        for path_id in frontiers_paths :
            norm_path_cost = norm(frontiers_paths[path_id]['aggregated_cost'])
            path_color = self.exploration_path_selection_cmap(norm_path_cost)
            path = frontiers_paths[path_id]
            if outline:
                outline_markers_array = getBubblesPathMarkerArray(path['path'], (0,0,0,1), self.exploration_path_selection_linewidth*1.8, 0, height_offset + 0.01*norm_path_cost - 0.01, show_nodes=False, lifetime=lifetime, id_offset=id_offset)
                id_offset = outline_markers_array.markers[-1].id + 1
                markers_list.extend(outline_markers_array.markers)
            path_markers_array = getBubblesPathMarkerArray(path['path'], path_color, self.exploration_path_selection_linewidth, 0, height_offset + 0.01*norm_path_cost, show_nodes=False, lifetime=lifetime, id_offset=id_offset)
            
            id_offset = path_markers_array.markers[-1].id + 1
            markers_list.extend(path_markers_array.markers)
            score_marker = getTextMarker(id_offset, path['path'].waypoints[-1], height_offset, "{:.3f}".format(norm_path_cost), path_color, size=self.exploration_path_selection_labels_size, pos_offset=np.array([0.2,0.2]), lifetime=lifetime)
            id_offset += 1
            markers_list.append(score_marker)
        self.path_nodes_viz_publisher.publish(MarkerArray(self.constructClearMsg(lifetime=lifetime)))
        self.path_selection_viz_publisher.publish(MarkerArray(self.constructClearMsg()))
        self.path_selection_viz_publisher.publish(MarkerArray(markers_list))

    def publishCurrTargetViz(self, target_pos, target_rad, height_offset=0):
        marker = self.circles_provider.getCircMarker(0, target_pos, target_rad, height_offset, self.exploration_curr_target_color, self.exploration_curr_target_linewidth, lifetime=0)
        self.current_target_viz_publisher.publish(MarkerArray([marker]))
        