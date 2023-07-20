import numpy as np
import rospy
#from sdg_roadmap.sdg_roadmap_strategy import *
from map_processing.map_processing_utils import *

from std_msgs.msg import Int32, Float32, Bool, Header
from geometry_msgs.msg import Point, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from skeleton_disk_graph_roadmap.srv import GetDiskGraph

from sdg_roadmap_server import SDGRoadmapServer

class SDGExplorationVisualizer:
    def __init__(self):
        # Publishers
        self.replan_pos_publisher = rospy.Publisher("~replan_pos_viz", MarkerArray, queue_size=1, latch=True)
        self.frontiers_plan_publisher = rospy.Publisher("~frontiers_plan_viz", MarkerArray, queue_size=1, latch=True)
        self.past_plans_publisher = rospy.Publisher("~past_plans_viz", MarkerArray, queue_size=1, latch=True)

        # Parameters (handle with a dict)
        self.replan_pos_color = np.array([0,225,25,126])/255
        self.replan_pos_diam = 0.5
        self.replan_pos_h = 1

        self.frontiers_paths_h = 1
        self.frontiers_paths_valid_color = [0.8,0.5,0,0.7]
        self.best_frontier_highlight_color = [0.9,0.2,0.2,0.8]
        self.frontiers_paths_lwidth = 0.1

        self.past_plans_h = 1
        self.past_plans_color = np.array([0,225,25,126])/255
        self.past_plans_lwidth = 0.02

    def constructReplanPosVizMsg(self, replan_pos):
        """TODO"""
        markers_list = []
        #color = self.main_color
        #color[3] = 0.5
        marker_index = 0
        for rpos in replan_pos:
            #rpos = pixPos_to_worldPos(f, zeroCell_pos, self.current_map_data.resolution)
            # if v.parent is not None :
            vertex_marker = Marker()
            marker_header = Header()
            marker_header.frame_id = "map"
            marker_header.stamp = rospy.Time(0)
            vertex_marker.header = marker_header
            vertex_marker.id = marker_index
            vertex_marker.type = 3  # cylinder
            vertex_marker.pose = Pose(
                Point(rpos[0], rpos[1], self.replan_pos_h), Quaternion(0, 0, 0, 1))
            #color = colors(1.*v.level/max_vertex_level)
            vertex_marker.color.r = self.replan_pos_color[0]
            vertex_marker.color.g = self.replan_pos_color[1]
            vertex_marker.color.b = self.replan_pos_color[2]
            vertex_marker.color.a = self.replan_pos_color[3]
            vertex_marker.scale.x = self.replan_pos_diam
            vertex_marker.scale.y = self.replan_pos_diam
            vertex_marker.scale.z = self.replan_pos_diam
            #vertex_marker.points = [Point(v.pos[0], v.pos[1], 0), Point(v.parent.pos[0], v.parent.pos[1], 0)]
            vertex_marker.lifetime = rospy.Duration(0)

            text_marker = Marker()
            text_marker.type = 9  # text
            text_marker.header = marker_header
            text_marker.color.r = self.replan_pos_color[0]
            text_marker.color.g = self.replan_pos_color[1]
            text_marker.color.b = self.replan_pos_color[2]
            text_marker.color.a = 1
            text_marker.scale.z = 0.3
            text_marker.pose = Pose(
                Point(rpos[0], rpos[1], 1.7*self.replan_pos_h), Quaternion(0, 0, 0, 1))
            text_marker.pose.position.x += self.replan_pos_diam
            text_marker.id = marker_index + 1
            text_marker.text = "rp_{}".format(int(marker_index/2))

            markers_list.append(vertex_marker)
            markers_list.append(text_marker)
            marker_index += 2
        marker_array = MarkerArray(markers_list)
        #color[3] = 1
        return marker_array

    def publishReplanPosViz(self, replan_pos):
        self.replan_pos_publisher.publish(self.constructReplanPosVizMsg(replan_pos))

    def constructFrontiersPlanVizMsg(self, frontiers_paths, best_path_id):
        """Builds a RViz visualization message for the planning phase to the graph frontier nodes"""
        markers_list = []
        marker_index = 0
        for path_id in frontiers_paths :
            path = frontiers_paths[path_id]
            #rpos = pixPos_to_worldPos(f, zeroCell_pos, self.current_map_data.resolution)
            # if v.parent is not None :
            vertex_marker = Marker()
            marker_header = Header()
            marker_header.frame_id = "map"
            marker_header.stamp = rospy.Time(0)
            vertex_marker.header = marker_header
            vertex_marker.id = marker_index
            vertex_marker.type = 4  # Line strip
            vertex_marker.points = [Point(p[0], p[1], 0) for p in path['path'].waypoints]
            vertex_marker.pose = Pose(
                Point(0,0,self.frontiers_paths_h), Quaternion(0, 0, 0, 1))
            #color = colors(1.*v.level/max_vertex_level)
            #if path_id in dist_valid_path_ids:
            #    path_color = self.frontiers_paths_valid_color
            #else:
            #    path_color = self.frontiers_paths_invalid_color
            path_color = self.frontiers_paths_valid_color
            vertex_marker.scale.x = self.frontiers_paths_lwidth
            if path_id == best_path_id:
                vertex_marker.scale.x *= 3
                path_color = self.best_frontier_highlight_color
            vertex_marker.color.r = path_color[0]
            vertex_marker.color.g = path_color[1]
            vertex_marker.color.b = path_color[2]
            vertex_marker.color.a = path_color[3]
            #vertex_marker.points = [Point(v.pos[0], v.pos[1], 0), Point(v.parent.pos[0], v.parent.pos[1], 0)]
            vertex_marker.lifetime = rospy.Duration(6)

            frontier_pos = path['path'].waypoints[-1]
            frontier_marker = Marker()
            frontier_marker.header = marker_header
            frontier_marker.id = marker_index + 1
            frontier_marker.type = 2  # sphere
            frontier_marker.pose = Pose(
                Point(frontier_pos[0], frontier_pos[1], self.frontiers_paths_h), Quaternion(0, 0, 0, 1))
            #color = colors(1.*v.level/max_vertex_level)
            frontier_marker.color.r = path_color[0]
            frontier_marker.color.g = path_color[1]
            frontier_marker.color.b = path_color[2]
            frontier_marker.color.a = path_color[3]
            frontier_marker.scale.x = 3*self.frontiers_paths_lwidth
            frontier_marker.scale.y = 3*self.frontiers_paths_lwidth
            frontier_marker.scale.z = 3*self.frontiers_paths_lwidth
            #vertex_marker.points = [Point(v.pos[0], v.pos[1], 0), Point(v.parent.pos[0], v.parent.pos[1], 0)]
            frontier_marker.lifetime = rospy.Duration(6)

            text_marker = Marker()
            text_marker.type = 9  # text
            text_marker.header = marker_header
            text_marker.color.r = path_color[0]
            text_marker.color.g = path_color[1]
            text_marker.color.b = path_color[2]
            text_marker.color.a = 1
            text_marker.scale.z = 0.5
            text_marker.pose = Pose(
                Point(frontier_pos[0], frontier_pos[1], 1.25*self.frontiers_paths_h), Quaternion(0, 0, 0, 1))
            text_marker.pose.position.x += 0.2
            text_marker.pose.position.y += 0.2
            text_marker.id = marker_index + 2
            text_marker.text = "{:.3f}".format(path['aggregated_cost'])
            text_marker.lifetime = rospy.Duration(6)

            markers_list.append(vertex_marker)
            markers_list.append(frontier_marker)
            markers_list.append(text_marker)
            marker_index += 3
        marker_array = MarkerArray(markers_list)
        return marker_array

    def publishFrontiersPlanViz(self, frontiers_paths, best_path_id):
        self.frontiers_plan_publisher.publish(self.constructFrontiersPlanVizMsg(frontiers_paths, best_path_id))

    def constructPastPlanVizMsg(self, past_plans_paths):
        """Builds a RViz visualization message for the planning phase to the graph frontier nodes"""
        markers_list = []
        marker_index = 0
        for path in past_plans_paths :
            #rpos = pixPos_to_worldPos(f, zeroCell_pos, self.current_map_data.resolution)
            # if v.parent is not None :
            vertex_marker = Marker()
            marker_header = Header()
            marker_header.frame_id = "map"
            marker_header.stamp = rospy.Time(0)
            vertex_marker.header = marker_header
            vertex_marker.id = marker_index
            vertex_marker.type = 4  # Line strip
            vertex_marker.points = [Point(p[0], p[1], 0) for p in path]
            vertex_marker.pose = Pose(
                Point(0,0,self.past_plans_h), Quaternion(0, 0, 0, 1))
            #color = colors(1.*v.level/max_vertex_level)
            path_color = self.past_plans_color
            vertex_marker.color.r = path_color[0]
            vertex_marker.color.g = path_color[1]
            vertex_marker.color.b = path_color[2]
            vertex_marker.color.a = path_color[3]
            vertex_marker.scale.x = self.past_plans_lwidth
            #vertex_marker.points = [Point(v.pos[0], v.pos[1], 0), Point(v.parent.pos[0], v.parent.pos[1], 0)]
            vertex_marker.lifetime = rospy.Duration(0)

            markers_list.append(vertex_marker)
            marker_index += 1
        marker_array = MarkerArray(markers_list)
        return marker_array

    def publishPastPlanViz(self, past_plans_paths):
        self.past_plans_publisher.publish(self.constructPastPlanVizMsg(past_plans_paths))
