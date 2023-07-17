from pqdict import pqdict
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time
from joblib import Parallel, delayed

from extended_mapping.map_processing import *
from extended_mapping.flux_skeletons_utils import *
from sdg_roadmap.graph_planner import *
from nav_utilities.paths import WaypointsPath

import json

import numpy.ma as ma
from skimage.draw import disk

class BubbleNode:
    """Class to represent a graph node as a free-space bubble

    Attributes:
        pos: array of size 2, 2D position of the node
        id: int, id of the node
        bubble_rad: float, free-space radius of the node
    """

    def __init__(self, pos, id):
        """Inits BubbleNode with pos and id"""
        self.pos = pos
        self.id = id
        self.bubble_rad = 0

    def __lt__(self, other):
        """Compares bubbles radii"""
        return self.bubble_rad < other.bubble_rad

    def setBubbleRad(self, bubble_rad):
        """Sets the bubble_rad attribute to a known value"""
        self.bubble_rad = bubble_rad

    def computeBubbleRad(self, dist_env_map, robot_rad=0.0):
        """Computes the bubble radius from the environment"""
        return dist_env_map.valueAt(self.pos) - robot_rad

    def updateBubbleRad(self, dist_env_map, robot_rad=0.0):
        self.setBubbleRad(self.computeBubbleRad(dist_env_map, robot_rad=robot_rad))


    def computeBubbleCoverage(self, occ_map, unkn_range=[-1,-1], inflation_rad_mult=1):
        bmask = circularMask(occ_map.dim, occ_map.worldToMapCoord(self.pos),
                                inflation_rad_mult*self.bubble_rad/occ_map.resolution)
        unkn_mask = (occ_map.data >= unkn_range[0])
        unkn_mask = unkn_mask*(occ_map.data <= unkn_range[1])
        unknown_in_bubble = unkn_mask*bmask
        #plt.imshow(np.logical_and(bmask, obst_mask))
        #plt.pause()
        return 1-np.sum(unknown_in_bubble)/np.sum(bmask), False


    def contains(self, pos, inflation=1.):
        """Checks if this bubble contains a given pos"""
        return np.linalg.norm(self.pos - pos) < inflation*self.bubble_rad 

    def edgeValidityTo(self, other_node, min_rad, res_tolerance=0):
        """Checks the strict edge validity condition to another node

        Args:
            other_node: BubbleNode, other node to which the edge is verified
            min_rad: float, minimal bubble radius

        Returns:
            A boolean, True if the edge is indeed valid
        """
        # TODO : ensure min_rad takes into account the correct distance (for now the definition is flawed a bit)
        return (self.bubble_rad + other_node.bubble_rad) > np.linalg.norm(self.pos - other_node.pos) - res_tolerance

    def betterEdgeValidityTo(self, other_node, min_rad, dist_map, res_tolerance=0):
        if self.edgeValidityTo(other_node, min_rad, res_tolerance=res_tolerance):
            p0, p1 = self.pos, other_node.pos
            r0, r1 = self.bubble_rad, other_node.bubble_rad
            d = np.linalg.norm(self.pos - other_node.pos)
            if d < r0 or d < r1 :
                return True 
            mid_pos = p0 + 0.5*(1+(r0 - r1)*(r0 + r1)/(d*d))*(p1 - p0) 
            if dist_map.valueAt(mid_pos) > min_rad:
                return True
            else:
                return False
        else:
            return False

    def refinedEdgeValidityTo(self, other_node, skeleton_dist_map):
        max_rad = max(self.bubble_rad, other_node.bubble_rad)
        min_x, max_x = min(self.pos[0], other_node.pos[0]) - max_rad, max(self.pos[0], other_node.pos[0]) + max_rad
        min_y, max_y = min(self.pos[1], other_node.pos[1]) - max_rad, max(self.pos[1], other_node.pos[1]) + max_rad
        bb_corners = np.array([[min_x, max_y],[max_x, min_y]])
        skel_in_bb, submap = skeletonInBoundingBox(skeleton_dist_map, bb_corners)
        self_mask = circularMask(submap.dim, submap.worldToMapCoord(self.pos), self.bubble_rad/submap.resolution)
        other_mask = circularMask(submap.dim, submap.worldToMapCoord(other_node.pos), other_node.bubble_rad/submap.resolution)
        overlap_mask = np.logical_and(self_mask, other_mask)
        return np.sum(np.multiply(skel_in_bb, overlap_mask))>0

    def old_filterPosListInd(self, pos_list, inner_range_mult=0, outer_range_mult=1):
        """Filters a list of positions to extract the ones in specified range

        Args:
            pos_list: list or array of shape (N,2), representing 2D positions
            inner_range_mult: radius multiplier defining the inner radius of the validity zone 
            outer_range_mult: radius multiplier defining the outer radius of the validity zone

        Returns:
            The indices of the valid positions in the list/array
        """
        
        pos_kdtree = KDTree(pos_list)
        if inner_range_mult > 0:
            inside_inner_ind = pos_kdtree.query_ball_point(
                self.pos, inner_range_mult*self.bubble_rad)
        else:
            inside_inner_ind = []
        inside_outer_ind = pos_kdtree.query_ball_point(
            self.pos, outer_range_mult*self.bubble_rad)
        return list(set(inside_outer_ind) - set(inside_inner_ind))
        
        #dists = np.linalg.norm(self.pos - np.array(pos_list), axis=1)
        #dists = cdist([self.pos], np.array(pos_list))[0]
        #inside_inner_ind = np.where(dists < self.bubble_rad*inner_range_mult)[0]
        #inside_outer_ind = np.where(dists < self.bubble_rad*outer_range_mult)[0]
        #return list(set(inside_outer_ind) - set(inside_inner_ind))

    def filterPosListInd(self, pos_list, inner_range_mult=0, outer_range_mult=1):
        """Filters a list of positions to extract the ones in specified range

        Args:
            pos_list: list or array of shape (N,2), representing 2D positions
            inner_range_mult: radius multiplier defining the inner radius of the validity zone 
            outer_range_mult: radius multiplier defining the outer radius of the validity zone

        Returns:
            The indices of the valid positions in the list/array
        """
        dists = distancesPointToSet(self.pos, pos_list)
        valid = np.where(np.logical_and(dists < outer_range_mult*self.bubble_rad,dists >= inner_range_mult*self.bubble_rad))
        return valid 

def segmentSkeletonJoints(skeleton_dist_map):
    skeleton_dist_map.data[np.where(skeleton_dist_map.data == 0)] = -1
    skeletal_points = np.array(np.where(skeleton_dist_map.data > 0)).T
    if not len(skeletal_points):
        print("no skel joints")
        return None, None
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

'''
def neighborsMap(skeleton_dist_map):
    #skeleton_dist_map.data[np.where(skeleton_dist_map.data == 0)] = -1
    #skeletal_points = np.array(np.where(skeleton_dist_map.data > 0)).T
    nb_coord = np.array([
        #[-1,-1],
        [-1,0],
        #[-1,1],
        [0,1],
        #[1,1],
        [1,0],
        #[1,-1],
        [0,-1]
    ])
    kernel = np.array([[-1, -1, -1],
                   [-1,  1, -1],
                   [-1, -1, -1]])
    nb_maps = [
        np.roll(np.roll(skeleton_dist_map.data, coord[0], axis=0), coord[1], axis=1) for coord in nb_coord
    ]
    return convolve(skeleton_dist_map.data, kernel)
'''

def getEdgesSkeletonMap(skeleton_dist_map, comp_size_filter=0):
    joints, skel = segmentSkeletonJoints(skeleton_dist_map)
    segmented_dist_map = skeleton_dist_map.copy()
    for p in joints :
        segmented_dist_map.data[p[0], p[1]] = -1

    cc_seg = cv2.connectedComponents((segmented_dist_map.data > 0).astype(np.uint8), connectivity=8)
    labels_map = cc_seg[1].copy()
    for i in range(np.max(labels_map)+1):
        if np.sum(labels_map == i) < comp_size_filter:
            labels_map[np.where(labels_map == i)] = 0
    return segmented_dist_map, labels_map


def getBubblesFromSkelCandidates(pos_candidates, dist_map, min_bubble_rad=0.5, d_offset=0, min_cov=0):
    bubbles = []
    closed_ind = []
    pq_dist = pqdict()
    pix_pos_cand = dist_map.worldToMapCoord(pos_candidates)
    cand_dists = [dist_map.data[p[0], p[1]] for p in pix_pos_cand]
    open_candidates = list(range(len(pos_candidates)))

    for i,p in enumerate(pos_candidates):
        pq_dist.additem(i, -cand_dists[i])
    
    #bubbles_mask = np.zeros(dist_map.dim)
    while len(pq_dist) > 0:
        #print(len(pq_dist))
        max_dist_ind = pq_dist.top()
        max_dist_px_pos = pix_pos_cand[max_dist_ind]
        if max_dist_ind in closed_ind :
            continue
        #if bubbles_mask[max_dist_px_pos[0], max_dist_px_pos[1]] == 0:
        if cand_dists[max_dist_ind] + d_offset > min_bubble_rad : 
            bn = BubbleNode(pos_candidates[max_dist_ind], len(bubbles))
            bn.updateBubbleRad(dist_map)
            bn.bubble_rad += d_offset

            #bn_mask = bubblesMask(dist_map.dim, [dist_map.worldToMapCoord(bn.pos)], [bn.bubble_rad/dist_map.resolution], radInflationMult=0.9)
            #bubbles_mask = np.logical_or(bubbles_mask, bn_mask)
            loc_open_candidates = [pos_candidates[i] for i in open_candidates]
            loc_closed_ind = bn.filterPosListInd(loc_open_candidates, inner_range_mult=0.0, outer_range_mult=1)
            #print(loc_closed_ind)
            loc_closed_ind = [open_candidates[i] for i in loc_closed_ind]
            for ci in loc_closed_ind :
                if ci in pq_dist:
                    pq_dist.pop(ci)
                closed_ind.append(ci)
                open_candidates.remove(ci)
            bubbles.append(bn)
        if not max_dist_ind in closed_ind:
            closed_ind.append(max_dist_ind)
            pq_dist.pop(max_dist_ind)
    return bubbles

def updatedBubblesFromSkel(pos_candidates, dist_map, min_bubble_rad, d_offset=0):
    eval_time = 0
    bubbles = []
    active_ind_mask = np.array([True]*len(pos_candidates), bool)
    pq_dist = pqdict()
    #pq_dist = []
    pix_pos_cand = dist_map.worldToMapCoord(pos_candidates)
    #cand_dists = [dist_map.data[p[0], p[1]] for p in pix_pos_cand]
    cand_dists = dist_map.data[pix_pos_cand[:,0], pix_pos_cand[:,1]]
    #cand_dists = [dist_map.valueAt(p) for p in pos_candidates]
    for i, p in enumerate(pos_candidates) :
        pq_dist.additem(i, -cand_dists[i])
    
    while len(pq_dist):
        active_ind = np.argwhere(active_ind_mask == True).flatten()
        max_dist_ind = pq_dist.top()        
        if not active_ind_mask[max_dist_ind]:
            continue
        # Check bubble validity condition
        
        if cand_dists[max_dist_ind] + d_offset > min_bubble_rad:
            # Add bubble
            bn = BubbleNode(pos_candidates[max_dist_ind], len(bubbles))
            bn.setBubbleRad(cand_dists[max_dist_ind] + d_offset)
            bubbles.append(bn)            
            # Close corresponding candidates
            #local_open_cand = [pos_candidates[i] for i in active_ind] #!!!!!!!! the performance increase ! *10 faster
            local_open_cand = pos_candidates[active_ind]
            local_closed_ind = bn.filterPosListInd(local_open_cand, inner_range_mult=0.0, outer_range_mult=0.9)            
            closed_ind = active_ind[local_closed_ind]
            for ci in closed_ind:
                #if ci in pq_dist:
                if active_ind_mask[ci]:
                    pq_dist.pop(ci)
                active_ind_mask[ci] = False
        active_ind_mask[max_dist_ind] = False
        if max_dist_ind in pq_dist:
            pq_dist.pop(max_dist_ind)
    #print("local open total: {}s".format(eval_time))
    return bubbles
             

def newBubblesFromSkel(dist_map, skeleton_map, min_bubble_rad, d_offset=0):
    """ NOT WORKING - TO IMPROVE"""
    JOINTS_FAVOR_FACTOR = 100
    bubbles = []
    skel_joints, skel_links = segmentSkeletonJoints(skeleton_map)
    skel_joints = skeleton_map.mapToWorldCoord(skel_joints)
    skel_links = skeleton_map.mapToWorldCoord(skel_links)
    #joints_dists = [dist_map.data[p[0], p[1]] for p in skel_joints]
    joints_dists = [dist_map.valueAt(p) for p in skel_joints]
    n_joints = len(joints_dists)
    #links_dists = [dist_map.data[p[0], p[1]] for p in skel_links]
    links_dists = [dist_map.valueAt(p) for p in skel_links]

    candidates = np.vstack([skel_joints,skel_links])
    cand_dists = joints_dists + links_dists
    #print(cand_dists)
    inter_cand_dists = np.array([float('inf')]*len(cand_dists))

    open_candidates = list(range(len(cand_dists)))
    pq_dist = pqdict()
    for i, c in enumerate(open_candidates):
        mult = 1
        if i < n_joints:
            mult = JOINTS_FAVOR_FACTOR
        pq_dist.additem(i, -cand_dists[i]*mult)
    
    while(len(pq_dist) > 0):
        print(len(pq_dist))
        max_dist_ind = pq_dist.top()
        if cand_dists[max_dist_ind] + d_offset > min_bubble_rad : 
            bn = BubbleNode(candidates[max_dist_ind], len(bubbles))
            bn.updateBubbleRad(dist_map)
            bn.bubble_rad += d_offset
            to_new_dists = np.array(distancesPointToSet(bn.pos, candidates[open_candidates])) - bn.bubble_rad
            loc_inter_cand_dists = np.minimum.reduce([np.array(inter_cand_dists[open_candidates]), np.array(to_new_dists)])
            loc_closed_ind = list(np.where(loc_inter_cand_dists < 0)[0])
            
            for ci in loc_closed_ind :
                #print(len(candidates))
                #print(ci)
                if ci in pq_dist:
                    pq_dist.pop(ci)
                    open_candidates.remove(ci)
                #inter_cand_dists.pop(ci
            bubbles.append(bn)
        if max_dist_ind in pq_dist:
            pq_dist.pop(max_dist_ind)
    return bubbles





def getJunctionBubbles(skeleton_dist_map, dist_map, min_bubble_rad = 0.5, d_offset=0):
    """
    TODO : docstring
    """
    t0 = time.time()
    joint_points, _ = segmentSkeletonJoints(skeleton_dist_map)
    t1 = time.time()
    print("\tskel_joints_seg: {}".format(t1 - t0))
    if not len(joint_points):
        return None
    scl_joint_points = dist_map.mapToWorldCoord(joint_points)
    #plt.scatter(scl_joint_points[:,0], scl_joint_points[:,1], c='red')
    #return getBubblesFromSkelCandidates(scl_joint_points, dist_map, min_bubble_rad=min_bubble_rad, d_offset=d_offset)
    return updatedBubblesFromSkel(scl_joint_points, dist_map, min_bubble_rad, d_offset=d_offset)


def getPatchingBubbles(masked_skel_map, dist_map, min_bubble_rad = 0.5, d_offset=0):
    """
    TODO : docstring
    """
    skel_points = np.array(np.where(masked_skel_map > 0)).T
    if skel_points is None :
        return None
    scl_skel_points = dist_map.mapToWorldCoord(skel_points)
    #return getBubblesFromSkelCandidates(scl_skel_points, dist_map, min_bubble_rad=min_bubble_rad, d_offset=d_offset)
    return updatedBubblesFromSkel(scl_skel_points, dist_map, min_bubble_rad, d_offset=d_offset)


def slow_circularMask(env_dim, center, radius):
    """Returns a circular boolean mask

    Args:
        envDim: list or array of size 2, describing the size of the grid environment
        center: list or array of size 2 representing the position of the circle center in the grid
        radius: float, radius of the wanted circular mask

    Returns:
        A 2D boolean-valued numpy array, with 1 inside the circle, 0 outside 
    """
    X, Y = np.ogrid[:env_dim[0], :env_dim[1]]
    #dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    #mask = dist_from_center <= radius
    dist_from_center = ((X - center[0])*(X - center[0]) + (Y-center[1])*(Y-center[1]))
    mask = dist_from_center <= radius*radius
    return mask


def circularMask(env_dim, center, radius):
    mask = np.zeros(env_dim, dtype=bool)
    rr, cc = disk(center, radius, shape=env_dim)
    mask[rr, cc] = 1
    return mask

def bubblesMask(env_dim, bubbles_pos, bubbles_rad, radInflationMult=1.0, radInflationAdd=0.0):
    """TODO: Docstring"""
    '''
    mask = np.zeros(env_dim)
    for k, bpos in enumerate(bubbles_pos):
        mask = np.logical_or(mask, circularMask(
            env_dim, bpos, bubbles_rad[k]*radInflationMult + radInflationAdd))
    '''

    '''
    masks = []
    for k, bpos in enumerate(bubbles_pos):
        masks.append(circularMask(env_dim, bpos, bubbles_rad[k]*radInflationMult + radInflationAdd))
    return np.logical_or.reduce(masks)
    '''
    mask = np.zeros(env_dim, dtype=bool)
    for k, bpos in enumerate(bubbles_pos):
        mask = np.logical_or(mask, circularMask(env_dim, bpos, bubbles_rad[k]*radInflationMult + radInflationAdd))
    return mask

def circDistMap(env_dim, center, radius, reverse=False):
    """Returns the euclidean distance field to the specified circle

    Args:
        envDim: list or array of size 2, describing the size of the grid environment
        center: list or array of size 2 representing the position of the circle center in the grid
        radius: float, radius of the circle the distance is computed to

    Returns:
        A 2D real-valued numpy array, representing the distance to the surface of the circle (i.e. 0 inside and on the perimeter)
    """
    if reverse : 
        rev_radius = radius
        radius = 0
    X, Y = np.ogrid[:env_dim[0], :env_dim[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    dist_map = (dist_from_center - radius)*(1.-(dist_from_center <= radius))
    if reverse:
        dist_map = (rev_radius - dist_map)*(dist_map < rev_radius)
    return dist_map


def bubblesDistanceField(env_dim, bubbles_pos, bubbles_rad):
    dist = []
    for k,bp in enumerate(bubbles_pos):
        dmap = circDistMap(env_dim, bp, bubbles_rad[k], reverse=True)
        dist.append(dmap)
    return np.maximum.reduce(dist)


def computeSplineControlPoints(init_path_pos, init_path_rad):
    m_list = []
    for k, path_vertex in enumerate(init_path_pos[:-1]) :
        t_b = min(1, init_path_rad[k]/np.linalg.norm(path_vertex - init_path_pos[k+1]))
        t_bnext = min(1,init_path_rad[k+1]/np.linalg.norm(path_vertex - init_path_pos[k+1]))
        m = path_vertex + (t_b - 0.5*(t_b + t_bnext - 1) )*(init_path_pos[k+1] - path_vertex)
        m_list.append(m)
    
    control_points = []
    b0 = init_path_pos[0]
    b0_control_points = [
        b0 - (m_list[0] - b0)/3,
        b0,
        b0 + (m_list[0] - b0)/3,
        b0 + (m_list[0] - b0)*2/3
    ]
    control_points.extend(b0_control_points)
    
    for k,path_vertex in enumerate(init_path_pos[1:-1]) :
        b = path_vertex
        b_ind = k+1
        b_control_points = [
            b + (m_list[b_ind - 1] - b)*2/3,
            b + (m_list[b_ind - 1] - b)/3,
            b + (m_list[b_ind - 1] - b)/8 + (m_list[b_ind] - b)/8,
            b + (m_list[b_ind] - b)/3,
            b + (m_list[b_ind] - b)*2/3
        ]
        control_points.extend(b_control_points)
    
    blast = init_path_pos[-1]   
    blast_control_points = [
        blast + (m_list[-1] - blast)*2/3,
        blast + (m_list[-1] - blast)/3,
        blast,
        blast - (m_list[-1] - blast)/3,
    ]
    control_points.extend(blast_control_points)
    return control_points

def computeSplinePath(control_points, n_interp=10):
    spline_vals = []
    for k, cp in enumerate(control_points) :
        if k==0 or k>len(control_points)-3 :
            continue
        if k == len(control_points) - 3:
            s = np.arange(n_interp)/(n_interp-1)
        else:
            s = np.arange(n_interp)/(n_interp)
        s = s.reshape(-1,1)
        loc_spline_vals = (1./6)*(
                np.multiply((1-s)*(1-s)*(1-s),control_points[k-1]) 
                + (3*s*s*s - 6*s*s + 4)*control_points[k]
                + (-3*s*s*s + 3*s*s + 3*s + 1)*control_points[k+1]
                + s*s*s*control_points[k+2])
        spline_vals.extend(loc_spline_vals)
        
    wp_path = WaypointsPath(np.array(spline_vals))
    return wp_path


def getSkelInteresections(bubble_nodes, skeleton_dist_map):
    bpos = [skeleton_dist_map.worldToMapCoord(b.pos) for b in bubble_nodes]
    brad = [b.bubble_rad/skeleton_dist_map.resolution for b in bubble_nodes]
    out_bmask = bubblesMask(skeleton_dist_map.dim, bpos, brad)
    in_bmask = bubblesMask(skeleton_dist_map.dim, bpos, brad, radInflationAdd=-3)
    b_edge_mask = out_bmask*(1-in_bmask)
    skel_intersections = np.array(np.where(b_edge_mask*skeleton_dist_map.data > 0)).T
    return skeleton_dist_map.mapToWorldCoord(skel_intersections)
    
class SkeletonDiskGraphTuningParameters:
    def __init__(self, parameters_dict):
        """
        Expects tuning parameters as a dictionary of floats and/or lists with keys :
            - "min_jbubbles_rad" : junction bubbles min radius (m)
            - "min_pbubbles_rad" : patching bubbles min radius (m)
            - "bubbles_dist_offset" : distance offset added to the bubbles radii (m)
            - "knn_checks" : number of graph nodes checked on k-nearest neighbors checks
            - "path_subdiv_length" : path_subdivision length for post-processing (m)
        """
        self.min_jbubbles_rad = parameters_dict["min_jbubbles_rad"]
        self.min_pbubbles_rad = parameters_dict["min_pbubbles_rad"]
        self.bubbles_dist_offset = parameters_dict["bubbles_dist_offset"]
        self.knn_checks = parameters_dict["knn_checks"]
        self.path_subdiv_length = parameters_dict["path_subdiv_length"]
        

class SkeletonDiskGraph(GraphPlanner):
    """Implements the skeleton planner"""
    def __init__(self, tuning_parameters):
        GraphPlanner.__init__(self)
        self.cached_dist_map = None
        self.kd_tree = None
        self.tuning_parameters = tuning_parameters
        self.last_added_ids = []
        #self.frozen_ids = []
        self.frontiers_ids = []

    @staticmethod
    def loadFromFile(filepath):
        loaded_data = json.load(open(filepath, 'r'))
        tuning_parameters = SkeletonDiskGraphTuningParameters(loaded_data['parameters'])
        
        planner = SkeletonDiskGraph(tuning_parameters)
        
        nodes_data = loaded_data['nodes']
        edges_data = loaded_data['edges']

        for nid in nodes_data:
            node = BubbleNode(np.array(nodes_data[nid]['pos']), int(nid))
            node.setBubbleRad(nodes_data[nid]['rad'])
            planner.addNode(node, override_id=False)

        for eid in edges_data:
            id0, id1 = edges_data[eid]['nodes']
            planner.addEdge(id0, id1, edges_data[eid]['cost'])

        return planner


    def updateSkelBubbles(self, dist_map, skeleton_dist_map): #, min_junction_bubbles_rad, min_patch_bubbles_rad, d_offset=0.):
        """
        Computes free space bubbles covering a skeleton map and the corresponding edges, in 2 passes
        First computes a mask of the current bubbles
        First computes bubbles at skeleton junctions (n_neighbors > 2)
        Then computes remaining bubbles after masking the skeleton
        """
        BUBBLES_RAD_PIX_OFFSET = 2
        t0_junctions = time.time()
        masked_skeleton_dist_map = skeleton_dist_map.copy()
        masked_skeleton_dist_map.setData(masked_skeleton_dist_map.data*(dist_map.data > self.tuning_parameters.min_pbubbles_rad))
        if len(self.graph.nodes):
            curr_bubbles_pos = [self.graph.nodes[bid]['node_obj'].pos for bid in self.graph.nodes]
            curr_bubbles_rad = [self.graph.nodes[bid]['node_obj'].bubble_rad for bid in self.graph.nodes]
            curr_bubbles_mask = bubblesMask(skeleton_dist_map.dim, skeleton_dist_map.worldToMapCoord(curr_bubbles_pos), np.array(curr_bubbles_rad)/dist_map.resolution, radInflationAdd = 0)
            masked_skeleton_dist_map.setData(masked_skeleton_dist_map.data*(1-curr_bubbles_mask))

        junction_bubbles = getJunctionBubbles(masked_skeleton_dist_map, dist_map, min_bubble_rad=self.tuning_parameters.min_jbubbles_rad, d_offset=self.tuning_parameters.bubbles_dist_offset)
        if junction_bubbles is not None:
            jpos = np.array([b.pos for b in junction_bubbles])
            jrad = np.array([b.bubble_rad for b in junction_bubbles])

        # Build mask
        jmasked_skeleton = masked_skeleton_dist_map.data
        if junction_bubbles is not None and len(junction_bubbles) :
            junction_pix_pos = masked_skeleton_dist_map.worldToMapCoord(jpos)
            junction_radii = jrad/dist_map.resolution #[jb.bubble_rad/dist_map.resolution for jb in junction_bubbles]
            junction_mask = bubblesMask(masked_skeleton_dist_map.dim, junction_pix_pos, junction_radii, radInflationAdd=-BUBBLES_RAD_PIX_OFFSET)
            jmasked_skeleton = masked_skeleton_dist_map.data*(1-junction_mask)
        t1_junctions = time.time()
        print("\tbubbles_joints_pass: {}".format(t1_junctions - t0_junctions))
        t0_patch = time.time()
        # Patching bubbles
        patch_bubbles = getPatchingBubbles((jmasked_skeleton > 0), dist_map, min_bubble_rad=self.tuning_parameters.min_pbubbles_rad, d_offset=self.tuning_parameters.bubbles_dist_offset)
        t1_patch = time.time()
        print("\tbubbles_pass2: {}".format(t1_patch - t0_patch))
        # Return all bubbles
        skel_bubbles = patch_bubbles
        if junction_bubbles is not None:
            skel_bubbles = junction_bubbles + skel_bubbles
        
        ### Visualization
        '''
        ppos = np.array([b.pos for b in patch_bubbles])
        prad = np.array([b.bubble_rad for b in patch_bubbles])
        # Junction bubbles
        skel_joints, other = segmentSkeletonJoints(masked_skeleton_dist_map)
        map_skel_joints = masked_skeleton_dist_map.mapToWorldCoord(skel_joints)
        plt.subplot(245)
        plt.title("Skeleton Joints")
        plt.scatter(map_skel_joints[:,0], map_skel_joints[:,1], c='green')
        masked_skeleton_dist_map.display(plt.cm.binary)
        
        plt.subplot(246)
        plt.title("Joints-Pass Bubbles")
        jmasked_skeleton_map = masked_skeleton_dist_map.copy()
        jmasked_skeleton[np.where(jmasked_skeleton==0)]=-1
        jmasked_skeleton_map.setData(jmasked_skeleton)
        jmasked_skeleton_map.display(plt.cm.binary)
        plt.scatter(jpos[:,0], jpos[:,1], c='green')
        for k,r in enumerate(jrad):
            circ = plt.Circle(jpos[k], r, ec='green', fc=(0,0,0,0), lw=2)
            plt.gca().add_patch(circ)

        plt.subplot(247)
        plt.title("2nd Pass Bubbles")
        jmasked_skeleton_map.display(plt.cm.binary)
        plt.scatter(ppos[:,0], ppos[:,1], c='green')
        for k,r in enumerate(prad):
            circ = plt.Circle(ppos[k], r, ec='green', fc=(0,0,0,0), lw=2)
            plt.gca().add_patch(circ)
        for k,r in enumerate(jrad):
            circ = plt.Circle(jpos[k], r, ec='darkblue', fc=(0,0,0,0), lw=0.8)
            plt.gca().add_patch(circ)

        plt.subplot(248)
        plt.title("Final Bubbles")
        masked_skeleton_dist_map.display(plt.cm.binary)
        for k, bub in enumerate(skel_bubbles):
            rad = bub.bubble_rad
            circ = plt.Circle(bub.pos, rad, ec='darkblue', fc=(0,0,0,0), lw=2)
            plt.gca().add_patch(circ)
        plt.show()
        '''
        return skel_bubbles

    '''
    def new_updateSkelBubbles(self, dist_map, occ_map, unkn_range, skeleton_dist_map): #, min_junction_bubbles_rad, min_patch_bubbles_rad, d_offset=0.):
        BUBBLES_RAD_PIX_OFFSET = 3
        masked_skeleton_dist_map = skeleton_dist_map.copy()
        if len(self.graph.nodes):
            curr_bubbles_pos = [self.graph.nodes[bid]['node_obj'].pos for bid in self.graph.nodes]
            curr_bubbles_rad = [self.graph.nodes[bid]['node_obj'].bubble_rad for bid in self.graph.nodes]
            curr_bubbles_mask = bubblesMask(dist_map.dim, dist_map.worldToMapCoord(curr_bubbles_pos), np.array(curr_bubbles_rad)/dist_map.resolution - BUBBLES_RAD_PIX_OFFSET)
            masked_skeleton_dist_map.setData(masked_skeleton_dist_map.data*(1-curr_bubbles_mask))
        
        junction_bubbles = getJunctionBubbles(masked_skeleton_dist_map, dist_map, occ_map, min_bubble_rad=self.tuning_parameters.min_jbubbles_rad, d_offset=self.tuning_parameters.bubbles_dist_offset)
        if junction_bubbles is not None:
            skel_intersections = getSkelInteresections(junction_bubbles, masked_skeleton_dist_map)
            #patch_bubbles = []
            #finished = False
            while (True):
                new_bubbles = getBubblesFromSkelCandidates(skel_intersections, dist_map, occ_map, unkn_range, min_bubble_rad=self.tuning_parameters.min_pbubbles_rad, d_offset=self.tuning_parameters.bubbles_dist_offset, min_cov=0)
                if not len(new_bubbles):
                    break
                junction_bubbles.extend(new_bubbles)
                curr_bubbles_pos = [nb.pos for nb in new_bubbles]
                curr_bubbles_rad = [nb.bubble_rad for nb in new_bubbles]
                curr_bubbles_mask = bubblesMask(dist_map.dim, dist_map.worldToMapCoord(curr_bubbles_pos), np.array(curr_bubbles_rad)/dist_map.resolution - BUBBLES_RAD_PIX_OFFSET)
                masked_skeleton_dist_map.setData(masked_skeleton_dist_map.data*(1-curr_bubbles_mask))
                skel_intersections = getSkelInteresections(junction_bubbles, masked_skeleton_dist_map)

        return junction_bubbles
    '''

    def updateSkelEdges(self, newly_added_ids, dist_map, robot_rad=0, res_tolerance=0): #, k_nn=5):
        checked_ids = []
        new_edges = []
        to_check = newly_added_ids.copy()

        while(len(to_check)):
            curr_id = to_check[0]
            curr_b = self.graph.nodes[curr_id]['node_obj']
            neighbors_ids = self.getClosestNodes(curr_b.pos, k=self.tuning_parameters.knn_checks+1)
            for bid in neighbors_ids:
                neighbor_b = self.graph.nodes[bid]['node_obj']
                if bid != curr_id :
                    #if curr_b.edgeValidityTo(neighbor_b, robot_rad):
                    if curr_b.betterEdgeValidityTo(neighbor_b, robot_rad, dist_map, res_tolerance=res_tolerance):
                        new_edges.append([curr_b.id, bid])
            to_check.pop(0)
        return new_edges


    def updateEdgesCost(self):
        max_cost = 0
        for e in self.graph.edges(data=True):
            b0_pos, b1_pos = self.graph.nodes[e[0]]['node_obj'].pos, self.graph.nodes[e[1]]['node_obj'].pos
            e_cost = np.linalg.norm(b0_pos - b1_pos)
            self.graph.edges[e[0], e[1]]['cost'] = e_cost
            #max_cost = max(e_cost, #)


    def updateOnDistMapUpdate(
        self, 
        new_dist_map, 
        new_skeleton_dist_map, 
        distance_change_update_thresh=0.,
        res_tolerance=0
    ):
        init_ids = list(self.graph.nodes)
        # Update current bubbles
        COMPONENTS_MIN_SIZE = 4
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
        skel_bubbles = self.updateSkelBubbles(new_dist_map, new_skeleton_dist_map) #, self.tuning_parameters.min_jbubbles_rad, self.tuning_parameters.min_pbubbles_rad, d_offset=d_offset)
        #skel_bubbles = self.new_updateSkelBubbles(new_dist_map, occ_map, new_skeleton_dist_map) #, self.tuning_parameters.min_jbubbles_rad, self.tuning_parameters.min_pbubbles_rad, d_offset=self.tuning_parameters.bubbles_dist_offset)
        new_ids = []
        for b in skel_bubbles:
            new_id = self.addNode(b)
            new_ids.append(new_id)
        t1_b = time.time()
        #print("Bubbles update : {} s".format(t1_b - t0_b))
        self.kd_tree = None
        # Compute new edges - TODO : update all edges costs separately
        t0_e = time.time()
        skel_edges = self.updateSkelEdges(new_ids, new_dist_map, robot_rad=self.tuning_parameters.min_pbubbles_rad, res_tolerance=res_tolerance)
        for e in skel_edges:
            b0_pos = self.graph.nodes[e[0]]['node_obj'].pos
            b1_pos = self.graph.nodes[e[1]]['node_obj'].pos
            self.addEdge(e[0], e[1], cost=np.linalg.norm(b0_pos - b1_pos))
        t1_e = time.time()
        print("edges_update: {}".format(t1_e - t0_e))
        # Remove small disconnected components
        self.removeSmallComponents(COMPONENTS_MIN_SIZE)
        # Update edges cost
        #self.updateEdgesCost()
        # Reset kd tree and cache distance map
        self.kd_tree = None
        self.cached_dist_map = new_dist_map.copy()

        self.last_added_ids = set(self.graph.nodes) - set(init_ids)
        #self.frozen_ids = set(self.graph.nodes) - self.last_added_ids
        #print("Full graph update : {} s".format((t1_e - t0_e) + (t1_b - t0_b)))
        return modified_ids


    def removeSmallComponents(self, min_comp_size):
        """Removes all edges inside connected components under a minimal size to allow for nodes removal

        Args:
            min_comp_size: int, minimal number of nodes in a connected component to avoid edges removal
        """
        invalid_edges = []
        for component in list(nx.connected_components(self.graph)):
            if (len(component) < min_comp_size):
                for node in component:
                    invalid_edges.extend(self.graph.edges(node))
        self.graph.remove_edges_from(invalid_edges)
        isolated = self.getIsolatedNodes()
        for ind in isolated:
            self.removeNode(ind)
    
    def getIsolatedNodes(self):
        """Returns IDs of all current graph nodes with no neighbors."""
        isolated = []
        for nid in self.graph.nodes:
            if not(len(self.graph.edges(nid))):
                isolated.append(nid)
        return isolated
    

    def getNodesKDTree(self):
        """Gets the scipy.KDTree derived from the current graph nodes"""
        if self.kd_tree is None:
            nodes_pos = [self.graph.nodes[nid]
                        ["node_obj"].pos for nid in self.nodes_ids]
            self.kd_tree = KDTree(nodes_pos)
        return self.kd_tree

    def getClosestNodes(self, pos, k=1):
        """Gets the k closest nodes, in the current graph, to a given position

        Args:
            pos: np.array of size 2, 2D position to compute the closest nodes to
            k: int, number of closest nodes to return

        Returns:
            A numpy arraay containing the k closest nodes IDs
        """
        if len(self.graph.nodes) == 0:
            return None
        k = min(k, len(self.graph.nodes))
        nodes_tree = self.getNodesKDTree()
        dist, ids = nodes_tree.query(pos, k=k)
        if k == 1:
            return [np.array(self.nodes_ids)[ids]]
        return np.array(self.nodes_ids)[ids]
    

    def isReachable(self, pos):
        """Returns True if the given pos is inside one of the current graph bubbles, and the ID of the corresponding node"""
        closest_ind = self.getClosestNodes(pos, k=self.tuning_parameters.knn_checks)
        closest = [self.graph.nodes[i]["node_obj"] for i in closest_ind]
        #closest = sorted(closest, reverse=False)
        local_rad = self.cached_dist_map.valueAt(pos)
        brads = np.array(
            [c.bubble_rad for c in closest])
        for k, cl in enumerate(closest):
            if np.linalg.norm(cl.pos - pos) < brads[k] + local_rad:
                return True, cl.id
        return False, -1

    
    def getNodesInRange(self, source_id, path_length_range):
        #all_paths = nx.all_shortest_paths(self.graph, source_id)
        in_range = []
        all_paths = nx.single_source_dijkstra_path_length(self.graph, source_id, weight='cost')
        for p in all_paths:
            if path_length_range[0] < all_paths[p] < path_length_range[1]:
                in_range.append(p)
        return in_range

    '''
    def getWorldPath(self, world_start, world_goal): #, postprocess_subd_length):
        """
        Gets a path between a start and a goal world pos
        Returns None if no path exists
        """
        start_reachable, start_id = self.isReachable(world_start)
        goal_reachable, goal_id = self.isReachable(world_goal)
        world_path = {}
        world_path['nodes_ids'] = None
        world_path['postprocessed'] = None
        if not (start_reachable and goal_reachable):
            print("Start reachable :{}\nGoal reachable :{}\nEither start or goal is unreachable in current graph".format(
                start_reachable, goal_reachable))
            return world_path
        if not nx.has_path(self.graph, start_id, goal_id):
            print("No path exists between start {} and goal {} in the current graph".format(
                start_id, goal_id))
            return world_path
        
        bubbles_path, bubbles_path_length = self.getPath(start_id, goal_id)
        pos_path, path_radii = self.postprocessPath(world_start, world_goal, bubbles_path, self.tuning_parameters.path_subdiv_length)
        world_path = {}
        world_path['nodes_ids'] = bubbles_path
        world_path['nodes_path_length'] = bubbles_path_length
        world_path['postprocessed'] = {'pos':pos_path, 'radii':path_radii}
        return world_path
    '''

    def hasWorldPath(self, world_start, world_goal):
        start_reachable, start_id = self.isReachable(world_start)
        goal_reachable, goal_id = self.isReachable(world_goal)
        if not (start_reachable and goal_reachable):
            return False
        return nx.has_path(self.graph, start_id, goal_id) 
    
    def getWorldPath(self, world_start, world_goal, postprocess=False): #, postprocess_subd_length):
        """
        Gets a path between a start and a goal world pos
        Returns None if no path exists
        """
        start_reachable, start_id = self.isReachable(world_start)
        goal_reachable, goal_id = self.isReachable(world_goal)
        if not (start_reachable and goal_reachable):
            return None, None, None
        world_path, world_path_length = self.getPath(start_id, goal_id)
        if world_path is None:
            return None, None, None
        if not postprocess:
            nodes_wp_path = [self.graph.nodes[ind]['node_obj'].pos for ind in world_path]
            world_wp_path = [world_start] + nodes_wp_path + [world_goal]
            world_path_length += np.linalg.norm(self.graph.nodes[world_path[0]]['node_obj'].pos - world_start)
            world_path_length += np.linalg.norm(self.graph.nodes[world_path[-1]]['node_obj'].pos - world_goal)
            return world_wp_path, world_path_length, world_path
        postprocessed = self.postprocessPath(world_start, world_goal, world_path)
        return postprocessed[0], WaypointsPath(np.array(postprocessed[0])).getTotalLength(), world_path

    def reduceBubblesPath(self, bubbles_pos, bubbles_rad): # , d_offset=0.):
        def _getNextValidInd(b0_ind, min_rad):
            next_valid_ind = b0_ind
            valid_cond = True
            while (next_valid_ind < len(bubbles_pos) - 1) and valid_cond:
                valid_overlap_cond = bubbles_rad[next_valid_ind]+bubbles_rad[b0_ind] > np.linalg.norm(bubbles_pos[next_valid_ind] - bubbles_pos[b0_ind]) + self.tuning_parameters.bubbles_dist_offset
                valid_rad_cond = bubbles_rad[next_valid_ind] > min_rad
                valid_cond = valid_overlap_cond #and valid_rad_cond
                if valid_cond:
                    next_valid_ind += 1
                else:
                    next_valid_ind -= 1
            return next_valid_ind
        
        if len(bubbles_pos) == 2:
            return bubbles_pos

        final_path = []
        prev_curr_id = 0
        curr_id = 0
        next_id = 1
        while(next_id > prev_curr_id and curr_id < len(bubbles_pos)):
            final_path.append(bubbles_pos[curr_id])
            next_id = _getNextValidInd(curr_id, self.tuning_parameters.min_pbubbles_rad)
            prev_curr_id = curr_id
            curr_id = next_id
        return final_path
        '''
        while(next_id < len(bubbles_pos)):
            bp = bubbles_pos[curr_id]
            br = bubbles_rad[curr_id]
            final_path.append(bp)
            #curr_id = curr_id  + 1
            next_id = curr_id #+ 1
            #if next_id > len(bubbles_pos) - 1:
            #    continue
            valid_cond = True #bubbles_rad[next_id]+br > np.linalg.norm(bubbles_pos[next_id] - bp) + d_offset
            while(valid_cond and next_id < len(bubbles_pos)-2):
                valid_cond = bubbles_rad[next_id]+br > np.linalg.norm(bubbles_pos[next_id] - bp) + d_offset
                next_id += 1
            curr_id = next_id
        #final_path.append(bubbles_pos[-1])
        return final_path
        '''

    def old_postprocessPath(self, world_start, world_goal, path_indices, subdivize=True): #, subd_length):
        """
        Postprocesses a list of bubbles ids to a waypoints path
        Ensures we dont go back through the centers of the last and first bubbles
        """
        path_disk_waypoints = [self.graph.nodes[pi]["node_obj"].pos for pi in path_indices]
        path_disk_radii = [self.graph.nodes[pi]["node_obj"].bubble_rad for pi in path_indices]
        # If going trough only 1 bubble, link start to goal
        if len(path_indices) == 1:
            path_waypoints = np.concatenate([
                world_start.reshape(1, 2),
                world_goal.reshape(1, 2)
            ])
            path_radii = [0, 0]
            return np.array(path_waypoints), np.array(path_radii)
        
        #path_disk_waypoints = reduceBubblesPath(path_disk_waypoints, path_disk_radii)
        '''
        else :
            wp_path = WaypointsPath(np.array([world_start] + path_disk_waypoints + [world_goal]))
            wp_rad = [self.cached_dist_map.valueAt(world_start)] + path_disk_radii + [self.cached_dist_map.valueAt(world_goal)]
            return wp_path.waypoints, wp_rad
        '''
        wp_path = WaypointsPath(np.array([world_start] + path_disk_waypoints + [world_goal]))
        # Subdivide path in segments of equal length
        if not subdivize:
            return wp_path.waypoints, np.array([self.cached_dist_map.valueAt(wp) for wp in wp_path.waypoints])
        n_subd = int(wp_path.getTotalLength()/self.tuning_parameters.path_subdiv_length)
        if n_subd > 2:
            wp_path = wp_path.getSubdivized(n_subd)
        return wp_path.waypoints, np.array([self.cached_dist_map.valueAt(wp) for wp in wp_path.waypoints])

    def postprocessPath(self, world_start, world_goal, path_indices):
        path_disk_waypoints = [world_start] + [self.graph.nodes[pi]["node_obj"].pos for pi in path_indices] + [world_goal]
        path_disk_waypoints = WaypointsPath(np.array(path_disk_waypoints))
        subdiv_n = int(path_disk_waypoints.getTotalLength()/(0.25*self.tuning_parameters.min_pbubbles_rad))
        path_disk_waypoints = path_disk_waypoints.getSubdivized(subdiv_n).waypoints
        #path_disk_radii = [self.cached_dist_map.valueAt(world_start)] + [self.graph.nodes[pi]["node_obj"].bubble_rad for pi in path_indices] + [self.cached_dist_map.valueAt(world_goal)]
        path_disk_radii = [self.cached_dist_map.valueAt(wp) for wp in path_disk_waypoints] 
        # If going trough only 1 bubble, link start to goal
        '''
        if len(path_indices) == 1:
            path_waypoints = np.concatenate([
                world_start.reshape(1, 2),
                world_goal.reshape(1, 2)
            ])
            path_radii = [0, 0]
            return np.array(path_waypoints), np.array(path_radii)
        '''
        reduced_path = self.reduceBubblesPath(path_disk_waypoints, path_disk_radii)
        reduced_path_radii = np.array([self.cached_dist_map.valueAt(wp) for wp in reduced_path])
        
        path_cpoints = computeSplineControlPoints(reduced_path, reduced_path_radii)
        spline_path = computeSplinePath(path_cpoints)
        return spline_path.waypoints, np.array([self.cached_dist_map.valueAt(wp) for wp in spline_path.waypoints])


                

    def getNodesPaths(self, start, ids=None, postprocess=True, verbose=False): #subd_length, 
        eval_time = 0
        nodes_paths = {}
        nodes_ids = self.graph.nodes
        if ids is not None:
            nodes_ids = ids
        start_reachable, start_id = self.isReachable(start)
        if not start_reachable:
            print("Start reachable :{}\nEither start or goal is unreachable in current graph".format(
                start_reachable))
            return None 

        for k, bid in enumerate(nodes_ids):
            goal_pos = self.graph.nodes[bid]['node_obj'].pos
            nodes_path_ids = self.getPath(start_id, bid)[0]
            path, length, nodes_ids = self.getWorldPath(start, goal_pos, postprocess=postprocess)
            nodes_paths[bid] = {}
            nodes_paths[bid]['postprocessed'] = np.array(path)
            nodes_paths[bid]['nodes_ids'] = nodes_path_ids
            nodes_paths[bid]['nodes_path_length'] = length
        '''            
        for k, bid in enumerate(nodes_ids):
            bnode = self.graph.nodes[bid]['node_obj']
            nodes_path, nodes_path_length = self.getPath(start_id, bid) 
            #world_path = self.getWorldPath(start, bnode.pos, subd_length)
            #if world_path['nodes_ids'] is None:    
            #    continue
            if nodes_path is None:
                continue
            if postprocess:
                t0 = time.time()
                pp_path, pp_path_radii = self.postprocessPath(start, self.graph.nodes[bid]['node_obj'].pos, nodes_path) #, subd_length)
                t1 = time.time()
                eval_time += t1 - t0
                reduced_path = self.reduceBubblesPath(pp_path, pp_path_radii)
                reduced_path_radii = np.array([self.cached_dist_map.valueAt(wp) for wp in reduced_path])
                
                path_cpoints = computeSplineControlPoints(reduced_path, reduced_path_radii)
                #t0 = time.time()
                spline_path = computeSplinePath(path_cpoints)
                #t1 = time.time()
                #eval_time += (t1 - t0)
                
                #path_cpoints = computeSplineControlPoints(pp_path, pp_path_radii)
                #spline_path = computeSplinePath(path_cpoints)
                nodes_paths[bid] = {}
                nodes_paths[bid]['postprocessed'] = np.array(spline_path.waypoints)
                nodes_paths[bid]['nodes_ids'] = nodes_path
                nodes_paths[bid]['nodes_path_length'] = nodes_path_length
            else:
                nodes_paths[bid] = {}
                nodes_paths[bid]['postprocessed'] = np.array([self.graph.nodes[bid]['node_obj'].pos for bid in nodes_path])
                nodes_paths[bid]['nodes_ids'] = nodes_path
                nodes_paths[bid]['nodes_path_length'] = nodes_path_length
            print("Computed {}/{} paths".format(k+1, len(nodes_ids)), end='\r')
        '''
        print("Postprocess - postproc total: {}s".format(eval_time))
        return nodes_paths

    
    def getMultiNodePath(self, start, ordered_ids):
        start_reachable, start_id = self.isReachable(start)
        if not start_reachable:
            print("Start reachable :{}\nEither start or goal is unreachable in current graph".format(
                start_reachable))
            return None 
        curr_id = start_id
        multinode_path = [curr_id]
        for next_id in ordered_ids:
            path, length = self.getPath(curr_id, next_id)
            if path is None:
                return None
            else:
                multinode_path.extend(path[1:])
            curr_id = next_id
        pp_path, pp_path_radii = self.postprocessPath(start, self.graph.nodes[ordered_ids[-1]]['node_obj'].pos, multinode_path, subdivize=False) #, subd_length)
        #reduced_path = self.reduceBubblesPath(pp_path, pp_path_radii)
        #reduced_path_radii = np.array([self.cached_dist_map.valueAt(wp) for wp in reduced_path])

        path_cpoints = computeSplineControlPoints(pp_path, pp_path_radii)
        spline_path = computeSplinePath(path_cpoints)
        final_path = {}
        final_path['postprocessed'] = np.array(spline_path.waypoints)
        final_path['nodes_ids'] = multinode_path
        return final_path        

    def segmentUnknown(self, occ_map, unkn_range, cov_threshold = 0):
        unkn_ids = []
        for bid in self.graph.nodes:
            bnode = self.graph.nodes[bid]['node_obj']
            cov, has_obst = bnode.computeBubbleCoverage(occ_map, unkn_range=unkn_range)
            if cov <= cov_threshold:
                unkn_ids.append(bid)
        return unkn_ids

    def segmentFrontiers(self, pos, occ_map, unkn_range, unkn_threshold=0, frontiers_threshold=0, max_search_dist = float('inf')):
        """
        WORK IN PROGRESS
        """
        MIN_FRONTIER_RAD_MULT = 1.

        checked_ids = []
        frontiers_ids = []
        coverages = {}

        #closest_ind = self.getClosestNodes(pos, k=self.tuning_parameters.knn_checks)[0]
        #curr_checking_range = 5

        #while not len(frontiers_ids) and len(checked_ids) < len(self.graph.nodes):
        init_node = self.getClosestNodes(pos, 1)[0]
        unkn_ids = self.getNodesInRange(init_node, [0,max_search_dist])
        #for bid in self.graph.nodes:
        for bid in unkn_ids:
            bnode = self.graph.nodes[bid]['node_obj']
            cov, has_obst = bnode.computeBubbleCoverage(occ_map, unkn_range=unkn_range)
            coverages[bid] = cov
            if cov <= unkn_threshold and bnode.bubble_rad > self.tuning_parameters.min_pbubbles_rad*MIN_FRONTIER_RAD_MULT:
                #unkn_ids.append(bid)
                for nid in self.graph.neighbors(bid):
                    frontiers_ids.append(nid)

        '''
        for bid in unkn_ids:
            graph_neighb_ids = [e[1] for e in self.graph.edges(bid)]
            added_neighb = False
            for nid in graph_neighb_ids:
                if coverages[nid] <= frontiers_threshold:
                    if nid not in unkn_ids and nid not in frontiers_ids:
                        frontiers_ids.append(nid)
                if coverages[nid] > frontiers_threshold:
                    frontiers_ids.append(bid)
                    break
                #else:
                #    if nid not in frontiers_ids:
                #        frontiers_ids.append(bid)
        '''
        return frontiers_ids

    def simpleFrontiersCharac(self, bubble_id, occ_map, unkn_range):
        bnode = self.graph.nodes[bubble_id]['node_obj']
        if unkn_range[0] < occ_map.valueAt(bnode.pos) < unkn_range[1]:
            #for nid in self.graph.neighbors(bubble_id):
            #    if occ_map.valueAt(self.graph.nodes[nid]['node_obj'].pos) < unkn_range[0]:
            return True
        return False

    def coverageFrontiersCharac(self, nodes_ids, occ_map, unkn_range, unkn_threshold):
        cand_frontiers = []
        frontiers = []
        frontiers_pos = []
        for bubble_id in nodes_ids:
            bnode = self.graph.nodes[bubble_id]['node_obj']
            cov, has_obst = bnode.computeBubbleCoverage(occ_map, unkn_range=unkn_range, inflation_rad_mult=0.9)
            if cov < unkn_threshold:
                cand_frontiers.append(bubble_id)
        for cand_id in cand_frontiers:
            for nid in self.graph.neighbors(cand_id):
                if not nid in cand_frontiers:
                    frontiers.append(cand_id)
                    frontier_dir = self.graph.nodes[cand_id]['node_obj'].pos - self.graph.nodes[nid]['node_obj'].pos
                    frontier_dir = frontier_dir/np.linalg.norm(frontier_dir)
                    frontier_pos = self.graph.nodes[cand_id]['node_obj'].pos - (self.graph.nodes[cand_id]['node_obj'].bubble_rad - self.graph.nodes[nid]['node_obj'].bubble_rad)*frontier_dir 
                    frontiers_pos.append(frontier_pos)
                    break
        return frontiers, frontiers_pos

    def searchClosestFrontiers(self, pos, occ_map, unkn_range, unkn_threshold, search_dist_increment, max_iter=100):
        init_node = self.getClosestNodes(pos, 1)[0]
        search_dist_range = [0, search_dist_increment]
        frontiers_ids = []
        curr_iter = 0
        while (not len(frontiers_ids)) and curr_iter < max_iter:
            check_ids = self.getNodesInRange(init_node, search_dist_range)
            #print(len(check_ids))
            '''
            #for bid in check_ids:
                #if self.simpleFrontiersCharac(bid, occ_map, unkn_range):
                #    frontiers_ids.append(bid)
            
                
                bnode = self.graph.nodes[bid]['node_obj']
                cov, has_obst = bnode.computeBubbleCoverage(occ_map, unkn_range=unkn_range)
                if cov < unkn_threshold and bnode.bubble_rad > self.tuning_parameters.min_pbubbles_rad:
                    for nid in self.graph.neighbors(bid):
                        nnode = bnode = self.graph.nodes[nid]['node_obj']
                        ncov, has_obst = nnode.computeBubbleCoverage(occ_map, unkn_range=unkn_range)
                        if ncov > 1 - unkn_threshold:
                            frontiers_ids.append(bid)
                            break
                    #frontiers_ids.append(bid)
            '''
            frontiers_ids, frontiers_pos = self.coverageFrontiersCharac(check_ids, occ_map, unkn_range, unkn_threshold)
            search_dist_range[0] += search_dist_increment
            search_dist_range[1] += search_dist_increment
            curr_iter += 1
        return frontiers_ids, frontiers_pos

    def saveToFile(self, filepath, parameters_dict):
        data_dict = {}
        #parameters = {}
        nodes_data = {}
        edges_data = {}
        for ind in self.graph.nodes:
            nodes_data[ind] = {}
            bpos = self.graph.nodes[ind]['node_obj'].pos
            brad = self.graph.nodes[ind]['node_obj'].bubble_rad
            nodes_data[ind]['pos'] = list(bpos)
            nodes_data[ind]['rad'] = brad
        for e in self.graph.edges:
            key = str(e[0])+"."+str(e[1])
            edges_data[key] = {}
            edges_data[key]['nodes'] = [int(e[0]), int(e[1])]
            #edges_data[e]['edge'] = e
            edges_data[key]['cost'] = self.graph.edges[e[0],e[1]]['cost']
        data_dict['parameters'] = parameters_dict
        data_dict['nodes'] = nodes_data
        data_dict['edges'] = edges_data
        json.dump(data_dict, open(filepath, 'w'))
        print("saved graph to {}".format(filepath))


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
                #plt.plot(edge_ends[:,0], edge_ends[:,1], c=edges_colors[k], lw=2, marker='.')
                #plt.plot(edge_ends[:,0], edge_ends[:,1], c=color, lw=elw)
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
                #circ = plt.Circle(subsampled_map.worldToMapCoord(b.pos), b.bubble_rad/subsampled_map.resolution, ec='red', fc=(0,0,0,0))
                #circ = plt.Circle(b.pos, b.bubble_rad, ec=bubble_colors[k], fc=(0,0,0,0))
                circ = plt.Circle(b.pos, b.bubble_rad, ec=color, fc=(0,0,0,0), lw=blw)
                plt.gca().add_patch(circ)
        if show_nodes:
            if nodes_color_dict is not None:
                plt.scatter(bpos[:,0], bpos[:,1], c=[nodes_color_dict[ind] for ind in nodes_color_dict], s=marker_size)
            else:
                plt.scatter(bpos[:,0], bpos[:,1], color=bcolor, s=marker_size)

    def displayNodesByDist(self, cmap=plt.cm.RdYlGn, hide_under_rad=0):
        skel_bubbles = [self.graph.nodes[ind]['node_obj'] for ind in self.graph.nodes]
        bpos = np.array([b.pos for b in skel_bubbles])
        brad = np.array([b.bubble_rad for b in skel_bubbles])
        for k,b in enumerate(skel_bubbles):
            if brad[k] > hide_under_rad:
                #color = bcolor
                #if b.id in highlight_ids:
                #    color=highlight_color
                #circ = plt.Circle(subsampled_map.worldToMapCoord(b.pos), b.bubble_rad/subsampled_map.resolution, ec='red', fc=(0,0,0,0))
                #circ = plt.Circle(b.pos, b.bubble_rad, ec=bubble_colors[k], fc=(0,0,0,0))
                norm_rad = (brad[k] - np.min(brad))/(np.max(brad) - np.min(brad))
                print(norm_rad)
                circ = plt.Circle(b.pos, b.bubble_rad, ec=cmap(norm_rad), fc=(0,0,0,0))
                plt.gca().add_patch(circ)
        #plt.scatter(bpos[:,0], bpos[:,1])

    def displayComponents(self):
        skel_bubbles = [self.graph.nodes[ind]['node_obj'] for ind in self.graph.nodes]
        bpos = np.array([b.pos for b in skel_bubbles])
        brad = np.array([b.bubble_rad for b in skel_bubbles])
        components = nx.connected_components(self.graph)
        for ci, c in enumerate(components):
            for k in c:
                b = self.graph.nodes[k]['node_obj']
                color = plt.cm.tab10(ci)
                circ = plt.Circle(b.pos, b.bubble_rad, ec=color, fc=(0,0,0,0))
                plt.gca().add_patch(circ)


