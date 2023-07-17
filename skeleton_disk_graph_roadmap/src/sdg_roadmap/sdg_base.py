import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from extended_mapping.flux_skeletons_utils import *

class GraphProvider:
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes_ids = []
        self.kd_tree = None

    # Utilities
    def getNewId(self):
        """Returns an unused (int) node ID"""
        if not len(self.nodes_ids):
            return 0
        return max(self.nodes_ids) + 1

    def addNode(self, new_node, override_id=True):
        """Adds a new node to the graph

        Args:
            node: the node to add, with an attribute id

        Returns:
            The ID of the added node, newly assigned when it gets added
        """
        if override_id:
            new_id = self.getNewId()
            new_node.id = new_id
        self.graph.add_node(new_node.id, node_obj=new_node)
        if not (new_node.id in self.nodes_ids):
            self.nodes_ids.append(new_node.id)
        return new_node.id

    def removeNode(self, node_id):
        """Removes a node from the graph.
        Its index also gets removed from the self.nodes_ids list.

        Args:
            node_id: int, the ID of the node to remove
        """
        self.graph.remove_node(node_id)
        self.nodes_ids.remove(node_id)

    def addEdge(self, id0, id1, cost=1.):
        """Adds a weighted edge to the graph"""
        self.graph.add_edge(id0, id1, cost=cost)

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

    def getPath(self, start_id, goal_id):
        """Returns the shortest path between a start and goal id
        Returns None if no path exists
        """
        if not nx.has_path(self.graph, start_id, goal_id):
            return None, None
        path = nx.shortest_path(
            self.graph, start_id, goal_id, weight="cost")
        length = nx.shortest_path_length(
            self.graph, start_id, goal_id, weight="cost")
        return path, length

    def getConnectedCompSubgraphs(self):
        """Returns all the connected components in the graph as subgraphs"""
        return [self.graph.subgraph(c).copy() for c in nx.connected_components(self.graph)]

    def getIsolatedNodes(self):
        """Returns IDs of all current graph nodes with no neighbors."""
        isolated = []
        for nid in self.graph.nodes:
            if not(len(self.graph.edges(nid))):
                isolated.append(nid)
        return isolated

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
        return 1-np.sum(unknown_in_bubble)/np.sum(bmask), False


    def contains(self, pos, inflation=1.):
        """Checks if this bubble contains a given pos"""
        return np.linalg.norm(self.pos - pos) < inflation*self.bubble_rad 

    def edgeValidityTo(self, other_node, res_tolerance=0):
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
        if self.edgeValidityTo(other_node, res_tolerance=res_tolerance):
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