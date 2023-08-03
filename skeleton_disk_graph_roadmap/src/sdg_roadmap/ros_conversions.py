import rospy
from skeleton_disk_graph_roadmap.msg import DiskGraph, DiskGraphNode, DiskGraphEdge
from sdg_roadmap.skel_disk_graph_provider import *

def skelDiskGraphToDiskGraphMsg(sdg_provider):
    """Builds and returns a skeleton_disk_graph_roadmap.msg/DiskGraph message representing the current graph state"""
    if not len(sdg_provider.graph.nodes):
        return
    else:
        disk_graph_msg = DiskGraph()
        disk_graph_msg.nodesIdsMapping = sdg_provider.nodes_ids
        disk_graph_msg_nodes = []
        nodes = [sdg_provider.graph.nodes[nid]["node_obj"]
                    for nid in sdg_provider.nodes_ids]
        #adj_matrix = np.zeros([len(nodes), len(nodes)], int)
        for node in nodes:
            graph_node = DiskGraphNode(
                node.id, node.pos[0], node.pos[1], node.bubble_rad)
            disk_graph_msg_nodes.append(graph_node)

        disk_graph_msg_edges = []
        for e in sdg_provider.graph.edges:
            edge = DiskGraphEdge(e[0], e[1])
            disk_graph_msg_edges.append(edge)
        disk_graph_msg.edges = list(disk_graph_msg_edges)
        disk_graph_msg.nodes = disk_graph_msg_nodes
        return disk_graph_msg

def diskGraphMsgToNodesAndEdges(disk_graph_msg):
    nodes_ids_mapping = np.array(disk_graph_msg.nodesIdsMapping)
    nodes_pos = np.array([[node.pos_x, node.pos_y] for node in disk_graph_msg.nodes])
    nodes_rad = [node.bubble_rad for node in disk_graph_msg.nodes]
    graph_nodes = [BubbleNode(p, nodes_ids_mapping[k]) for k, p in enumerate(nodes_pos)]
    graph_edges = [[e.n0_id, e.n1_id] for e in disk_graph_msg.edges]
    return graph_nodes, graph_edges