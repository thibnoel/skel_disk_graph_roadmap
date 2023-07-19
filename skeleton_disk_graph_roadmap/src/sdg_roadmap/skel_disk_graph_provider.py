### Skeleton Disk-Graph Provider : methods and class related to the construction of the roadmap graph

from pqdict import pqdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json
import time

from extended_mapping.map_processing import *
from sdg_roadmap.sdg_base import *
from sdg_roadmap.bubbles_paths import *


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
    """Extract the map coordinates of the skeleton joints in a binary map"""
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


class SkeletonDiskGraphProvider(GraphProvider):
    """
    Implements the skeleton disk-graph construction
    """
    def __init__(self, parameters_dict):
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
        GraphProvider.__init__(self)
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
        Gets a BubblesPath between a start and a goal world pos
        Returns None if no path exists
        """
        start_reachable, start_id = self.isGraphReachable(world_start)
        goal_reachable, goal_id = self.isGraphReachable(world_goal)
        if not (start_reachable and goal_reachable):
            return None
        nodes_path, nodes_path_length = self.getPath(start_id, goal_id)
        if nodes_path is None:
            return None
        bubble_nodes = [self.graph.nodes[ind]['node_obj'] for ind in nodes_path]
        node_start = BubbleNode(world_start, -1)
        node_start.updateBubbleRad(self.cached_dist_map)
        node_goal = BubbleNode(world_goal, -1)
        node_goal.updateBubbleRad(self.cached_dist_map)
        bubble_nodes = [node_start] + bubble_nodes + [node_goal]
        return BubblesPath([b.pos for b in bubble_nodes], [b.bubble_rad for b in bubble_nodes])
    
    def getWorldPathFromIds(self, start_id, goal_id):
        """
        Gets a BubblesPath between two specified ids
        Returns None if no path exists
        """
        nodes_path, nodes_path_length = self.getPath(start_id, goal_id)
        if nodes_path is None:
            return None
        bubble_nodes = [self.graph.nodes[ind]['node_obj'] for ind in nodes_path]
        return BubblesPath([b.pos for b in bubble_nodes], [b.bubble_rad for b in bubble_nodes])
    

    def getNodesInRange(self, source_id, path_length_range):
        """
        Returns all nodes ids reachable from source_id with a path of length in the specified range
        """
        in_range = []
        all_paths = nx.single_source_dijkstra_path_length(self.graph, source_id, weight='cost')
        for p in all_paths:
            if path_length_range[0] < all_paths[p] < path_length_range[1]:
                in_range.append(p)
        return in_range


    ##############################
    ### FRONTIERS CHARACTERIZATION FOR EXPLORATION

    def isFrontier(self, cand_node_id, occupancy_map, map_unkn_range, frontier_unkn_threshold):
        """
        Implements the frontier check described in the paper
        Checks 2 conditions :
            - candidate node covrage < frontier unknown coverage threshold
            - at least one neighbor centered in known free space
        """
        cand_bubble = self.graph.nodes[cand_node_id]['node_obj']
        cand_coverage = cand_bubble.computeBubbleCoverage(occupancy_map, unkn_range=map_unkn_range, inflation_rad_mult=0.9)
        if cand_coverage > frontier_unkn_threshold:
            return False
        for neighb_id in self.graph.neighbors(cand_node_id):
            neighb_node = self.graph.nodes[neighb_id]['node_obj']
            neighb_center_occ = occupancy_map.valueAt(neighb_node.pos)
            if neighb_center_occ < map_unkn_range[0]:
                return True
        return False

    def searchClosestFrontiers(self, source_pos, occupancy_map, unkn_range, unkn_threhsold, search_dist_increment, max_iter=100):
        """
        Implements frontier search from a source position in an incremental manner
        """
        init_node = self.getClosestNodes(source_pos, 1)[0]
        search_dist_range = [0, search_dist_increment]
        frontiers_ids = []
        frontiers_pos = []
        curr_iter = 0
        while (not len(frontiers_ids)) and curr_iter < max_iter:
            check_ids = self.getNodesInRange(init_node, search_dist_range)
            for cid in check_ids:
                if self.isFrontier(cid, occupancy_map, unkn_range, unkn_threhsold):
                    frontiers_ids.append(cid)
                    frontiers_pos.append(self.graph.nodes[cid]['node_obj'].pos)
            search_dist_range[0] += search_dist_increment
            search_dist_range[1] += search_dist_increment
            curr_iter += 1
        return frontiers_ids, frontiers_pos


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


class SDGExplorationPathProvider:
    def __init__(self, sdg_provider, map_unkn_occ_range, frontiers_max_known_ratio, frontiers_search_dist_increment):
        self.sdg_provider = sdg_provider
        self.map_unkn_occ_range = map_unkn_occ_range
        self.frontiers_max_known_ratio = frontiers_max_known_ratio
        self.frontiers_search_dist_increment = frontiers_search_dist_increment

    def computePathCosts(self, path, occupancy_map):
        """
        Computes the evaluation costs associated to a path
        THIS SHOULD BE THE ONLY METHOD TO CHANGE IF NEW SELECTION COSTS ARE ADDED
        """
        path_costs = {}
        path_costs['energy_penalty'] = path.getTotalLength()
        path_costs['coverage_reward'] = pathUnknownCoverageReward(path, occupancy_map, unkn_range=self.map_unkn_occ_range)
        return path_costs 

    def normalizeCosts(self, frontiers_paths_dict):
        """
        Takes a set of evaluated paths as a dictionary and normalizes all costs between 0 and 1
        """
        extracted_costs = {}
        for f_id in frontiers_paths_dict:
            extracted_costs[f_id] = frontiers_paths_dict[f_id]['costs']
        aggregated_costs = {}
        for cost in extracted_costs[list(extracted_costs.keys())[0]].keys():
            aggregated_costs[cost] = []
            for f_id in extracted_costs:    
                aggregated_costs[cost].append(extracted_costs[f_id][cost])
        normalizers = {}
        for cost in aggregated_costs:
            normalizers[cost] = {}
            normalizers[cost]['min'] = np.min(aggregated_costs[cost])
            normalizers[cost]['max'] = np.max(aggregated_costs[cost])

        for f_id in frontiers_paths_dict:
            for cost in frontiers_paths_dict[f_id]['costs']:
                frontiers_paths_dict[f_id]['costs'][cost] = (frontiers_paths_dict[f_id]['costs'][cost] - normalizers[cost]['min'])/(normalizers[cost]['max'] - normalizers[cost]['min'])

    def getFrontiersPaths(self, source_pos, occupancy_map):
        """
        Extracts frontiers from the Skeleton Disk-Graph, computes the associated paths and their evaluation costs
        Returns the result as a dict indexed by the id of the corresponding frontier node
        """
        source_id = self.sdg_provider.getClosestNodes(source_pos, 1)[0]
        frontiers_ids, _ = self.sdg_provider.searchClosestFrontiers(source_pos, occupancy_map, self.map_unkn_occ_range, self.frontiers_max_known_ratio, self.frontiers_search_dist_increment, max_iter=100)
        frontiers_paths = {}
        for f_id in frontiers_ids:
            path = self.sdg_provider.getWorldPathFromIds(source_id, f_id)
            if path is not None:
                frontiers_paths[f_id] = {}
                frontiers_paths[f_id]['path'] = path
                frontiers_paths[f_id]['costs'] = self.computePathCosts(path, occupancy_map)
        if len(frontiers_paths):
            self.normalizeCosts(frontiers_paths)
        return frontiers_paths

    def selectExplorationPath(self, frontiers_paths_dict, cost_param_dict):
        path_scores = {}
        for f_id in frontiers_paths_dict:
            path_score = 0
            for cost in frontiers_paths_dict[f_id]['costs']:
                path_score += frontiers_paths_dict[f_id]['costs'][cost]*cost_param_dict[cost]
            path_scores[f_id] = path_score
        best_path_id = max(path_scores, key=path_scores.get)
        return frontiers_paths_dict[best_path_id]