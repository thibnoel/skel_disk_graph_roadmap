from map_processing.flux_skeletons_utils import *
import networkx as nx
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class GraphPlanner:
    """Generic and minimal graph planner
    Holds the state of the graph, stores nodes and weighted edges
    Basically a wrapper for nx.Graph, specialized for path planning
    """

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

    
    def getWorldPath(self, start_pos, goal_pos):
        start_id = self.getClosestNodes(start_pos, k=1)[0]
        goal_id = self.getClosestNodes(goal_pos, k=1)[0]
        nodes_path, length = self.getPath(start_id, goal_id)
        if nodes_path is None:
            return None, None
        world_path = [self.graph.nodes[ind]['node_obj'].pos for ind in nodes_path]
        return world_path, length       


    def getConnectedCompSubgraphs(self):
        """Returns all the connected components in the graph as subgraphs"""
        return [self.graph.subgraph(c).copy() for c in nx.connected_components(self.graph)]

    def display(self, ecolor='red', bcolor='red', elw=1):
        nodes = [self.graph.nodes[ind]['node_obj'] for ind in self.graph.nodes]
        edges = self.graph.edges
        marker_size = 20
        plot_edges = []
        for k,e in enumerate(edges):
            b0 = [b for b in nodes if b.id == e[0]][0]
            b1 = [b for b in nodes if b.id == e[1]][0]
            color = ecolor
            edge_ends = np.array([b0.pos, b1.pos])
            #plt.plot(edge_ends[:,0], edge_ends[:,1], c=color, lw=elw)
            plot_edges.append(edge_ends)
        edge_lines = LineCollection(plot_edges, color=ecolor, lw=elw)
        plt.gca().add_collection(edge_lines)
        bpos = np.array([b.pos for b in nodes])
        plt.scatter(bpos[:,0], bpos[:,1], c=bcolor, s=marker_size)



    