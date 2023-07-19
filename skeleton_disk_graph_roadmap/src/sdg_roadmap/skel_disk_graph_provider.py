### Skeleton Disk-Graph Provider : methods and class related to the construction of the roadmap graph

from pqdict import pqdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

from extended_mapping.map_processing import *
from extended_mapping.geom_processing import *

from sdg_roadmap.graph_planner import *
from sdg_roadmap.sdg_base import *
from sdg_roadmap.bubbles_paths import BubblesPath

import json

import numpy.ma as ma



def bubblesMask(ref_env_map, bubbles, radInflationMult=1.0, radInflationAdd=0.0):
    """TODO: Docstring"""
    mask = np.zeros(ref_env_map.dim, dtype=bool)
    bpos = ref_env_map.worldToMapCoord([b.pos for b in bubbles])
    brad = np.array([b.bubble_rad for b in bubbles])/ref_env_map.resolution
    for k, b in enumerate(bubbles):
        mask = np.logical_or(mask, circularMask(ref_env_map.dim, bpos[k], brad[k]*radInflationMult + radInflationAdd))
    return mask

def computeValidBubbles(pos_candidates, dist_map, min_bubble_rad, dist_offset):
    """Implements the radius-descending bubbles extraction on a set of candidates"""
    bubbles = []
    active_ind_mask = np.array([True]*len(pos_candidates), bool)
    pq_dist = pqdict()
    pix_pos_cand = dist_map.worldToMapCoord(pos_candidates)
    cand_dists = dist_map.data[pix_pos_cand[:,0], pix_pos_cand[:,1]]
    for i, p in enumerate(pos_candidates) :
        pq_dist.additem(i, -cand_dists[i])
    
    while len(pq_dist):
        active_ind = np.argwhere(active_ind_mask == True).flatten()
        max_dist_ind = pq_dist.top()        
        if not active_ind_mask[max_dist_ind]:
            continue
        # Check bubble validity condition
        if cand_dists[max_dist_ind] + dist_offset > min_bubble_rad:
            # Add bubble
            bn = BubbleNode(pos_candidates[max_dist_ind], len(bubbles))
            bn.setBubbleRad(cand_dists[max_dist_ind] + dist_offset)
            bubbles.append(bn)            
            # Close corresponding candidates
            local_open_cand = pos_candidates[active_ind]
            local_closed_ind = bn.filterPosListInd(local_open_cand, inner_range_mult=0.0, outer_range_mult=0.9)            
            closed_ind = active_ind[local_closed_ind]
            for ci in closed_ind:
                if active_ind_mask[ci]:
                    pq_dist.pop(ci)
                active_ind_mask[ci] = False
        active_ind_mask[max_dist_ind] = False
        if max_dist_ind in pq_dist:
            pq_dist.pop(max_dist_ind)
    return bubbles

def segmentSkeletonJoints(skeleton_dist_map):
    skeleton_dist_map.data[np.where(skeleton_dist_map.data == 0)] = -1
    skeletal_points = np.array(np.where(skeleton_dist_map.data > 0)).T
    if not len(skeletal_points):
        print("no skel")
        return [], []
    joint_points = []
    pure_skeletal_points = []
    for pix_coord in skeletal_points:
        neighbors = getNeihgborsCoord(pix_coord)
        invalid_ind = []
        nb_val_neighb = 0
        for k,p in enumerate(neighbors):
            if p[0] < 0 or p[0] > skeleton_dist_map.dim[0] -1 or p[1] < 0 or p[1] > skeleton_dist_map.dim[1] -1 :
                invalid_ind.append(k)
        for k,p in enumerate(neighbors) :
            if k not in invalid_ind:
                if (skeleton_dist_map.data[p[0], p[1]] > 0):
                    #local_graph_nodes.append(p)
                    nb_val_neighb += 1
        if nb_val_neighb > 2:
            joint_points.append(pix_coord)
        else:
            pure_skeletal_points.append(pix_coord)        
    return np.array(joint_points), np.array(pure_skeletal_points)

def extractBubblesFromSkel(skeleton_map, dist_map, min_bubble_rad, dist_offset, from_skel_joints=False):
    source_points = None
    if from_skel_joints:
        joints, others = segmentSkeletonJoints(skeleton_map)
        source_points = joints
    else: 
        source_points = np.array(np.where(skeleton_map.data > 0)).T
    if not len(source_points):
        return []
    source_points = dist_map.mapToWorldCoord(source_points)
    return computeValidBubbles(source_points, dist_map, min_bubble_rad, dist_offset)


"""
Implements the skeleton disk-graph construction
"""
class SkeletonDiskGraphProvider(GraphProvider):
    """
    Initialization
        Args:
            - parameters_dict: dictionary formatted as :
                {
                    "min_jbubbles_rad": x,
                    "min_pbubbles_rad": x,
                    "bubbles_dist_offset": x,
                    "knn_checks": x
                }
    """
    def __init__(self, parameters_dict):
        GraphPlanner.__init__(self)
        self.cached_dist_map = None
        self.kd_tree = None
        self.parameters_dict = parameters_dict

    ##############################
    ### GRAPH CONSTRUCTION

    def computeUpdatedBubbleNodes(self, dist_map, skeleton_map, rad_pix_offset=2):
        """
        Update the current graph nodes
            Args:
                - dist_map: distance map extracted after preprocessing
                - skeleton_map: skeleton map extracted from the distance gradient map
                - rad_pix_offset: pixel offset used when computing the junctions bubbles mask
        """
        # Masking the skeleton with current existing bubbles
        masked_skeleton_map = skeleton_map.copy()
        masked_skeleton_map.setData(masked_skeleton_map.data*(dist_map.data > self.parameters_dict["min_pbubbles_rad"]))
        if len(self.graph.nodes):
            curr_nodes = [self.graph.nodes[bid]["node_obj"] for bid in self.graph.nodes]
            curr_nodes_mask = bubblesMask(dist_map, curr_nodes)
            masked_skeleton_map.setData(masked_skeleton_map.data*(1 - curr_nodes_mask))
        # Joints bubbles pass
        joints_bubbles = extractBubblesFromSkel(masked_skeleton_map, dist_map, self.parameters_dict["min_jbubbles_rad"], self.parameters_dict["bubbles_dist_offset"], from_skel_joints=True)
        joints_masked_skeleton_map = masked_skeleton_map.copy()  
        if len(joints_bubbles):
            joints_bubbles_mask = bubblesMask(masked_skeleton_map, joints_bubbles, radInflationAdd=-rad_pix_offset)      
            joints_masked_skeleton_map.setData(masked_skeleton_map.data*(1 - joints_bubbles_mask))    
        # Second pass
        other_bubbles = extractBubblesFromSkel(joints_masked_skeleton_map, dist_map, self.parameters_dict["min_pbubbles_rad"], self.parameters_dict["bubbles_dist_offset"], from_skel_joints=False)
        return joints_bubbles + other_bubbles

    def computeUpdatedEdges(self, new_ids, dist_map, min_rad, res_tolerance=0):
        """
        Update the current graph edges
        """
        new_edges = []
        to_check = new_ids.copy()

        while(len(to_check)):
            curr_id = to_check[0]
            curr_b = self.graph.nodes[curr_id]['node_obj']
            neighbors_ids = self.getClosestNodes(curr_b.pos, k=self.parameters_dict["knn_checks"]+1)
            for bid in neighbors_ids:
                neighbor_b = self.graph.nodes[bid]['node_obj']
                if bid != curr_id :
                    if curr_b.betterEdgeValidityTo(neighbor_b, min_rad, dist_map, res_tolerance=res_tolerance):
                        new_edges.append([curr_b.id, bid])
            to_check.pop(0)
        return new_edges

    
    def updateOnDistMapUpdate(self, new_dist_map, new_skeleton_map, distance_change_update_thresh=0, res_tolerance=0, min_comp_size=4):
        init_ids = list(self.graph.nodes)
        # Update current bubbles
        modified_ids = []
        for b_id in self.graph.nodes:
            bnode = self.graph.nodes[b_id]['node_obj']
            old_rad = bnode.bubble_rad
            new_rad = bnode.computeBubbleRad(new_dist_map)
            if np.abs(new_rad - old_rad) > distance_change_update_thresh:
                modified_ids.append(bnode.id)
        for ind in modified_ids:
            self.removeNode(ind)
        # Compute missing bubbles
        t0_b = time.time()
        skel_bubbles = self.computeUpdatedBubbleNodes(new_dist_map, new_skeleton_map)
        new_ids = []
        for b in skel_bubbles:
            new_id = self.addNode(b)
            new_ids.append(new_id)
        t1_b = time.time()
        #print("Bubbles update : {} s".format(t1_b - t0_b))
        self.kd_tree = None
        # Compute new edges - TODO : update all edges costs separately
        t0_e = time.time()
        skel_edges = self.computeUpdatedEdges(new_ids, new_dist_map, self.parameters_dict["min_pbubbles_rad"], res_tolerance=res_tolerance)
        for e in skel_edges:
            b0_pos = self.graph.nodes[e[0]]['node_obj'].pos
            b1_pos = self.graph.nodes[e[1]]['node_obj'].pos
            self.addEdge(e[0], e[1], cost=np.linalg.norm(b0_pos - b1_pos))
        t1_e = time.time()
        #print("edges_update: {}".format(t1_e - t0_e))
        # Remove small disconnected components
        self.removeSmallComponents(min_comp_size)

        # Reset kd tree and cache distance map
        self.kd_tree = None
        self.cached_dist_map = new_dist_map.copy()

        #print("Full graph update : {} s".format((t1_e - t0_e) + (t1_b - t0_b)))
        return modified_ids

    ##############################
    ### PATH PLANNING

    def isGraphReachable(self, world_pos):
        """Returns True if the given pos is reachable from one of the current graph bubbles, and the ID of the corresponding node"""
        closest_ind = self.getClosestNodes(world_pos, k=self.parameters_dict["knn_checks"])
        closest = [self.graph.nodes[i]["node_obj"] for i in closest_ind]
        #closest = sorted(closest, reverse=False)
        local_rad = self.cached_dist_map.valueAt(world_pos)
        brads = np.array([c.bubble_rad for c in closest])
        for k, cl in enumerate(closest):
            if np.linalg.norm(cl.pos - world_pos) < brads[k] + local_rad:
                return True, cl.id
        return False, -1

    def getWorldPath(self, world_start, world_goal):
        """
        Gets a path between a start and a goal world pos
        Returns None if no path exists
        """
        start_reachable, start_id = self.isGraphReachable(world_start)
        goal_reachable, goal_id = self.isGraphReachable(world_goal)
        if not (start_reachable and goal_reachable):
            return None
        nodes_path, nodes_path_length = self.getPath(start_id, goal_id)
        if nodes_path is None:
            return None
        #nodes_waypoints = [self.graph.nodes[ind]['node_obj'].pos for ind in nodes_path]
        bubble_nodes = [self.graph.nodes[ind]['node_obj'] for ind in nodes_path]
        node_start = BubbleNode(world_start, -1)
        node_start.updateBubbleRad(self.cached_dist_map)
        node_goal = BubbleNode(world_goal, -1)
        node_goal.updateBubbleRad(self.cached_dist_map)
        #nodes_waypoints = [world_start] + nodes_waypoints + [world_goal]
        bubble_nodes = [node_start] + bubble_nodes + [node_goal]
        #return WaypointsPath(nodes_waypoints)
        return BubblesPath([b.pos for b in bubble_nodes], [b.bubble_rad for b in bubble_nodes])

    ##############################
    ### GRAPH VISUALIZATION

    def display(self, show_bubbles, show_nodes, show_edges, bcolor='blue', ecolor='blue', blw=1, elw=1, highlight_ids=[], highlight_color='red', nodes_color_dict=None):
        skel_bubbles = [self.graph.nodes[ind]['node_obj'] for ind in self.graph.nodes]
        skel_edges = self.graph.edges
        plot_edges = []
        marker_size = 20
        if show_edges:
            for k,e in enumerate(skel_edges):
                b0 = [b for b in skel_bubbles if b.id == e[0]][0]
                b1 = [b for b in skel_bubbles if b.id == e[1]][0]
                color = ecolor
                if b0.id in highlight_ids or b1.id in highlight_ids:
                    color=highlight_color
                edge_ends = np.array([b0.pos, b1.pos])
                plot_edges.append(edge_ends)
        plt.gca().add_collection(LineCollection(plot_edges, color=color, lw=elw))
        bpos = np.array([b.pos for b in skel_bubbles])
        if show_bubbles:
            for k,b in enumerate(skel_bubbles):
                color = bcolor
                if nodes_color_dict is not None:
                    color = nodes_color_dict[b.id]
                if b.id in highlight_ids:
                    color=highlight_color
                circ = plt.Circle(b.pos, b.bubble_rad, ec=color, fc=(0,0,0,0), lw=blw)
                plt.gca().add_patch(circ)
        if show_nodes:
            if nodes_color_dict is not None:
                plt.scatter(bpos[:,0], bpos[:,1], c=[nodes_color_dict[ind] for ind in nodes_color_dict], s=marker_size)
            else:
                plt.scatter(bpos[:,0], bpos[:,1], color=bcolor, s=marker_size)