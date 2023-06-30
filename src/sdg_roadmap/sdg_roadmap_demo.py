import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize

from map_processing.map_processing_utils import *
from map_processing.farthest_point_sampling import *
from sdg_roadmap.sdg_roadmap_utils import *
from sdg_roadmap.sdg_roadmap_strategy import *

rcParams.update({'font.size': 24})

### Main steps :
# Map processing
    # Pre processing : obstacles extraction / binarization
    # Pre processing : occupancy dilation + erosion (smoothing effect)
    # Euclidean Distance Transform (EDT) : obtain SDF and gradient
    # SDF gradiet flux computation
    # Thin skeleton extraction (thresholding + thinning)
    
# Disk-Graph construction
    # Skeleton joints extraction
    # Placement of junction bubbles
    # Placement of patching bubbles

# Disk-Graph exploitation
    # Single-query path planning in known maps
    # Frontiers charatcerization and exploration planning in partially unknown maps

def preprocFigure(obst_map, dil_erode_thresholds):
    plt.subplot(121)
    plt.title("Obstacles Bin. Map")
    obst_map.display(cmap=plt.cm.binary)

    cumul_filtered_map = obst_map.copy()
    cumul_filtered_map.data = np.zeros(obst_map.data.shape) #cumul_filtered_map.data.astype(float)
    for t in dil_erode_thresholds:
        filtered_map = mapDilateErode(obst_map, t)
        cumul_filtered_map.data[np.where(cumul_filtered_map.data == 0)] = np.maximum(t*filtered_map.data, cumul_filtered_map.data)[np.where(cumul_filtered_map.data == 0)]
    cumul_filtered_map.data[np.where(cumul_filtered_map.data == 0)] = None
    plt.subplot(122)
    plt.title("Dilation + Erosion")
    cumul_filtered_map.display(cmap=plt.cm.hot)
    plt.colorbar(fraction=0.05, use_gridspec=False, ticks=dil_erode_thresholds[0::4],shrink=0.5, label="$d_{ed}$")

def distanceFigure(map_preproc_config, distance_map, cmap, contour_color=None, contour_lw=2, title=True):
    if title:
        plt.title("Signed Distance Field\nd_offset = {}".format(map_preproc_config['dist_field_offset']))
    distance_map.display(cmap=cmap)
    if contour_color is not None:
        X,Y = distance_map.getNpMeshgrid()
        plt.contour(X, Y, distance_map.data.T, levels=[0], colors=contour_color, linewidths=contour_lw)

def distanceGradFigure(map_preproc_config, distance_gradient_maps):
    # color inds : R:0, G:1, B:2
    x_color_ind = 1
    y_color_ind = 2 #B

    #x_color = [0.7,0.3,0.2]
    #y_color = [0.2,0.7,0.3]
    x_color = [.8,.2,.2]
    y_color = [.2,.8,.2]
    #x_color = [.5,.5,.5]
    #y_color = [-.5,-.5,-.5]

    grad_x, grad_y = distance_gradient_maps[0].data, distance_gradient_maps[1].data
    grad_array = np.array([grad_x, grad_y])
    xnorm = Normalize(np.min(grad_x), np.max(grad_x))
    ynorm = Normalize(np.min(grad_y), np.max(grad_y))
    x_vals = plt.cm.binary(xnorm(grad_x))
    y_vals = plt.cm.binary(ynorm(grad_y))
    grad_colors = np.zeros(x_vals.shape)
    grad_colors[:,:,:] = 0
    #grad_colors[:,:,x_color_ind] = x_vals[:,:,0]
    #grad_colors[:,:,y_color_ind] = y_vals[:,:,0]
    grad_colors[:,:,0] += x_vals[:,:,0]*x_color[0]
    grad_colors[:,:,1] += x_vals[:,:,0]*x_color[1]
    grad_colors[:,:,2] += x_vals[:,:,0]*x_color[2]
    grad_colors[:,:,0] += y_vals[:,:,0]*y_color[0]
    grad_colors[:,:,1] += y_vals[:,:,0]*y_color[1]
    grad_colors[:,:,2] += y_vals[:,:,0]*y_color[2]

    #grad_colors = 1-grad_colors

    no_grad_ind = np.array(np.where(dist_map.data == 0))
    grad_colors[no_grad_ind[0], no_grad_ind[1]] = 0
    grad_colors[:,:,3] = 1
    
    plt.title("SDF Gradient")
    distance_gradient_maps[0].display()
    plt.imshow(np.flip(grad_colors.transpose(1,0,2), axis=0), extent=distance_gradient_maps[0].getExtent(transpose=False))

def graphFigure(source_map, sdg_planner):
    graph_edges_color = 'darkred'
    graph_bubbles_color = 'red'

    plt.title("Skeleton Disk-Graph Roadmap")
    source_map.display(cmap=plt.cm.binary)
    sdg_planner.display(True, True, bcolor=graph_bubbles_color, ecolor=graph_edges_color, elw=2, blw=1)


def frontiersExtractionFigure(source_map, start, frontiers_paths, paths_cmap=plt.cm.cool):
    marker_size = 40
    linewidth = 3
    source_map.display(cmap=plt.cm.binary)
    max_l = np.max([frontiers_paths[p]['nodes_path_length'] for p in frontiers_paths])
    for k, path_id in enumerate(frontiers_paths):
        color = paths_cmap(frontiers_paths[path_id]['nodes_path_length']/max_l)
        path_poses = frontiers_paths[path_id]['postprocessed']
        plt.plot(path_poses[:,0], path_poses[:,1], c=color, lw=linewidth, zorder=100000-frontiers_paths[path_id]['nodes_path_length'])
        plt.scatter(path_poses[-1,0], path_poses[-1,1], c=color, s=marker_size)
    plt.scatter(start[0], start[1], c='green', zorder=10, s=2*marker_size)

def frontiersCostsFigure(source_map, start, frontiers_paths, path_costs, combin_weigths, paths_cmap=plt.cm.cool):
    marker_size = 40
    linewidth = 3
    for i, cost in enumerate(path_costs[list(path_costs.keys())[0]]):
        plt.subplot(2,len(path_costs[list(path_costs.keys())[0]].keys()), i+1)
        plt.title(cost)
        source_map.display(cmap=plt.cm.binary)
        max_c = np.max([path_costs[p][cost] for p in frontiers_paths])
        for k, path_id in enumerate(frontiers_paths):
            color = paths_cmap(path_costs[path_id][cost]/max_c)
            path_poses = frontiers_paths[path_id]['postprocessed']
            plt.plot(path_poses[:,0], path_poses[:,1], c=color, lw=linewidth, zorder=100000-path_costs[path_id][cost])
            plt.scatter(path_poses[-1,0], path_poses[-1,1], c=color, s=marker_size)

    plt.subplot(2,1,2)
    source_map.display(cmap=plt.cm.binary)
    plt.title("Combined cost ({})".format(combin_weigths))
    tot_costs = {}
    for k, path_id in enumerate(frontiers_paths):
        tot_cost = 0
        for i, cost in enumerate(path_costs[list(path_costs.keys())[0]]):
            tot_cost += combin_weigths[i]*path_costs[path_id][cost]
        tot_costs[path_id] = tot_cost
    for k, path_id in enumerate(frontiers_paths):
        max_c = np.max([tot_costs[p] for p in frontiers_paths])
        min_c = np.min([tot_costs[p] for p in frontiers_paths])
        color = paths_cmap((tot_costs[path_id] - min_c)/(max_c - min_c))
        path_poses = frontiers_paths[path_id]['postprocessed']
        plt.plot(path_poses[:,0], path_poses[:,1], c=color, lw=linewidth, zorder=100000-tot_cost)
        plt.scatter(path_poses[-1,0], path_poses[-1,1], c=color, s=marker_size)
    plt.scatter(start[0], start[1], c='green', zorder=10, s=2*marker_size)

def graphDistFigure(source_map, source_id, sdg_planner, dist_step=10, cmap=plt.cm.viridis):
    scatt_size = 50
    checked_ids = []
    grouped_nodes = []
    dist_range = np.array([0, dist_step])
    while len(checked_ids) < len(sdg_planner.graph.nodes):
        reachable = sdg_planner.getNodesInRange(source_id, dist_range)
        if len(reachable) == 0:
            break
        grouped_nodes.append(reachable)
        checked_ids.extend(reachable)
        dist_range = dist_range + dist_step
        print(dist_range, len(checked_ids))
    
    source_map.display(cmap=plt.cm.binary)
    sdg_planner.display(False, True, ecolor='grey', elw=1)
    source_pos = sdg_planner.graph.nodes[source_id]['node_obj'].pos
    plt.scatter(source_pos[0], source_pos[1], color='red', s=2*scatt_size, zorder=10)
    for k, group in enumerate(grouped_nodes):
        c = cmap(k/(len(grouped_nodes) - 1))
        gnodes_pos = np.array([sdg_planner.graph.nodes[ind]['node_obj'].pos for ind in group])
        plt.scatter(gnodes_pos[:,0], gnodes_pos[:,1], color=c, s=scatt_size, zorder=10)
    return grouped_nodes

def graphAllPathsFigure(source_map, source_pos, sdg_planner, graph_color, nodes_ids=None, precomp_paths=None, cmap=plt.cm.viridis):
    #plt.title("Planning to all reachable nodes")
    #start_pos = sdg_planner.graph.nodes[source_id]['node_obj'].pos
    paths_gen_time = 0
    if precomp_paths is None :
        t0 = time.time()
        paths = sdg_planner.getNodesPaths(source_pos, ids=nodes_ids, postprocess=False)
        t1 = time.time()
        paths_gen_time += (t1 - t0)
    else:
        paths = precomp_paths
    print("Planning paths took {}s, avg = {}s/path".format(paths_gen_time, paths_gen_time/len(paths)))
    max_l = np.max([paths[p]['nodes_path_length'] for p in paths])
    source_map.display(cmap=plt.cm.binary)
    #sdg_planner.display(False, True, ecolor=graph_color, elw=1)
    for p in paths:
        postprocessed_path = paths[p]['postprocessed']
        path_length = paths[p]['nodes_path_length']
        plt.plot(postprocessed_path[:,0], postprocessed_path[:,1], c=cmap(path_length/max_l), zorder=100000-path_length, lw=3)
    return paths

maps_path_prefix = '/root/catkin_ws/src/ros_explore_mapping/src/map_processing/'
# Demonstrate in various maps with parameters for each

# CONFIGURATION : initial map file
map_file_config = {
    'map_path': "sim_map_lidar.pgm",
    'map_resolution': 0.05,
    'map_origin': np.array([0,0])
}
map_file_config = {
    'map_path': "sim_map_lidar2.pgm",
    'map_resolution': 0.05,
    'map_origin': np.array([0,0])
}
'''
map_file_config = {
    'map_path': "cmu_indoors_partial.pgm",
    'map_resolution': 0.1,
    'map_origin': np.array([0,0])
}

map_file_config = {
    'map_path': "map2.pgm",
    'map_resolution': 0.1,
    'map_origin': np.array([0,0])
}
'''

map_file_config = {
    'map_path': "env_intellab.png",
    'map_resolution': 0.1,
    'map_origin': np.array([0,0])
}



'''
map_file_config = {
    'map_path': "env_div.png",
    'map_resolution': 0.2,
    'map_origin': np.array([0,0])
}
'''
'''
map_file_config = {
    'map_path': "env_maze.jpeg",
    'map_resolution': 0.2,
    'map_origin': np.array([0,0])
}
'''
'''
map_file_config = {
    'map_path': "partial_maze.png",
    'map_resolution': 0.2,
    'map_origin': np.array([0,0])
}
'''
'''
map_file_config = {
    'map_path': "env_empty.png",
    'map_resolution': 0.2,
    'map_origin': np.array([0,0])
}
'''
'''
map_file_config = {
    'map_path': "custom_env_partial_map.pgm",
    'map_resolution': 0.2,
    'map_origin': np.array([0,0])
}
'''


### Map initialization
map_data = imageToArray(maps_path_prefix + map_file_config['map_path'])
env_map = EnvironmentMap(
    map_file_config['map_resolution'],
    map_data.shape,
    map_file_config['map_origin'],
    data = map_data
)
#env_map = extractSubMap(env_map, [0,25], [25,0])

# CONFIGURATION : map preprocessing
map_preproc_config = {
    'subsampling_factor': 1.,
    'obst_thresh': 0.4,
    'dilation_erosion_dist': 0.6,
    'dist_field_offset': 0.2
}
### Map preprocessing
# Subsampling
subsamp_factor = 1
subsampled_map = subsampleMap(env_map, map_preproc_config['subsampling_factor'])
# Obstacles extraction
obst_map = subsampled_map.copy()
obst_map.setData(subsampled_map.data > map_preproc_config['obst_thresh'])
# Smoothing (dilation + erosion) of obstacles
filtered_bin_map = mapDilateErode(obst_map, map_preproc_config['dilation_erosion_dist'])
filtered_occ_map = subsampled_map.copy()
filtered_occ_map.data[np.where(filtered_bin_map.data == 1)] = 1
# Signed distance field and gradients
dist_map, grad_maps = computeDistsScipy(filtered_bin_map, map_preproc_config['obst_thresh'], obst_d_offset=map_preproc_config['dist_field_offset'], compute_negative_d=True)
dist_grad_array = np.array([grad_maps[0].data, grad_maps[1].data])

# CONFIGURATION : Skeleton extraction - only 1 config. param, the flux threhsold
skeleton_extraction_config = {
    'flux_threshold': -1e-2
}
### Skeleton extraction
# Compute gradient flux
dist_grad_flux = gradFlux(dist_grad_array, include_diag_neighbors=True)
grad_flux_map = dist_map.copy()
grad_flux_map.setData(dist_grad_flux)

thresh_flux_map = grad_flux_map.copy()
thresh_flux_map.setData(thresh_flux_map.data < skeleton_extraction_config['flux_threshold'])

# Threhsold and thin the flux to obtain the skeleton
thin_flux = fluxToSkeletonMap(dist_grad_flux, skeleton_extraction_config['flux_threshold'])
skeleton_map = subsampled_map.copy()
skeleton_map.setData(thin_flux.astype(int)*(dist_map.data > 0.2))


# Dist to thin skeleton

plt.figure()
#plt.subplot(1,4,1)
env_map.display(plt.cm.binary)
#plt.suptitle("From Distance To Thin Skeleton")
plt.figure()
plt.subplot(241)
plt.title("SDF")
distanceFigure(map_preproc_config, dist_map, plt.cm.RdYlGn, contour_color='blue', contour_lw=3, title=False)
plt.subplot(242)
plt.title("Grad. Flux")
grad_flux_map.display(cmap=plt.cm.binary)
plt.subplot(243)
plt.title("Thresh. Flux")
thresh_flux_map.display(cmap=plt.cm.binary)
plt.subplot(244)
plt.title("Thin Skel.")
skeleton_map.display(cmap=plt.cm.binary)
#plt.show()



# CONFIGURATION : Disk-graph construction
sdg_construction_config = {
    "min_jbubbles_rad" : 0.6,
    "min_pbubbles_rad" : 0.4,
    "bubbles_dist_offset" : 0.,
    "knn_checks" : 100,
    "path_subdiv_length" : 0.4
}
### Disk-graph construction
# Use dedicated parameters class to initialize planner
sdg_tuning_param = SkeletonDiskGraphTuningParameters(sdg_construction_config)
sdg_planner = SkeletonDiskGraph(sdg_tuning_param)
# Update the planner using the skeleton map
sdg_planner.updateOnDistMapUpdate(dist_map, skeleton_map)
#exit()
### Exploration
sdg_strategy_config = {
    "min_frontier_unkn_ratio": 0,
    "max_frontier_unkn_ratio": 0,
    "unkn_occ_range": [0.1,0.6], 
    "narrowness_cost_d_threshold": 0, 
    "max_dist_travel_compromise": 5,
    "unknown_max_coverage": 0.4,
    "frontiers_min_coverage": 0.2
}
# Strategy definition
sdg_strategy = SDGFrontiersStrategy(
    sdg_planner,
    sdg_strategy_config["min_frontier_unkn_ratio"],
    sdg_strategy_config["max_frontier_unkn_ratio"],
    sdg_strategy_config["unkn_occ_range"],
    sdg_strategy_config["narrowness_cost_d_threshold"],
    sdg_strategy_config["max_dist_travel_compromise"],
    sdg_strategy_config["unknown_max_coverage"],
    sdg_strategy_config["frontiers_min_coverage"],
)
# Frontiers extraction
#start = np.array([12,20])
#valid_frontier_ids, valid_frontiers_paths = sdg_strategy.newGetValidFrontiers(start, filtered_occ_map, dist_map, 100)

#path_costs = sdg_strategy.getFrontiersPathsCosts(valid_frontier_ids, valid_frontiers_paths, filtered_occ_map)

#plt.figure()
preprocFigure(obst_map, np.linspace(0,3,21))
plt.show()

plt.figure()
plt.subplot(121)
distanceFigure(map_preproc_config, dist_map, plt.cm.RdYlGn, contour_color='blue', contour_lw=3)
plt.subplot(122)
distanceGradFigure(map_preproc_config, grad_maps)
plt.show()
exit()

occ_lims = [0.1,0.6]

n_samples = 100000
initial_points = np.array(np.where(skeleton_map.data > 0)).T
#initial_points = np.array(np.where(subsampled_map.data[initial_points[:,0], initial_points[:,1]] < 0.6))
print(initial_points)
initial_points = subsampled_map.resolution*(initial_points) 

# Farthest point sampling
'''
t0 = time.time()
fps_samples, sep_dist = farthestPointSampling(initial_points, n_samples, target_sep_dist=1.6)
t1 = time.time()
print("FPS : {} s".format(t1 - t0))
#skeleton_map.display(cmap=plt.cm.binary)
subsampled_map.display(cmap=plt.cm.binary)
plt.scatter(fps_samples[:,0], fps_samples[:,1])
plt.show()
'''


#frontiersExtractionFigure(subsampled_map, start, valid_frontiers_paths)

combin_weights = [-3,.5,1.5]
#frontiersCostsFigure(subsampled_map, start, valid_frontiers_paths, path_costs, combin_weights, paths_cmap=plt.cm.cool_r) 
#plt.show()



### Map update
'''
env_map.data[100:300,100:300] = 0
# Subsampling
subsamp_factor = 1
subsampled_map = subsampleMap(env_map, map_preproc_config['subsampling_factor'])
# Obstacles extraction
obst_map = subsampled_map.copy()
obst_map.setData(subsampled_map.data > map_preproc_config['obst_thresh'])
# Smoothing (dilation + erosion) of obstacles
filtered_bin_map = mapDilateErode(obst_map, map_preproc_config['dilation_erosion_dist'])
filtered_occ_map = subsampled_map.copy()
filtered_occ_map.data[np.where(filtered_bin_map.data == 1)] = 1
# Signed distance field and gradients
dist_map, grad_maps = computeDistsScipy(filtered_bin_map, map_preproc_config['obst_thresh'], obst_d_offset=map_preproc_config['dist_field_offset'], compute_negative_d=True)
dist_grad_array = np.array([grad_maps[0].data, grad_maps[1].data])

# CONFIGURATION : Skeleton extraction - only 1 config. param, the flux threhsold
skeleton_extraction_config = {
    'flux_threshold': -1e-2
}
### Skeleton extraction
# Compute gradient flux
dist_grad_flux = gradFlux(dist_grad_array, include_diag_neighbors=True)
grad_flux_map = dist_map.copy()
grad_flux_map.setData(dist_grad_flux)

thresh_flux_map = grad_flux_map.copy()
thresh_flux_map.setData(thresh_flux_map.data < skeleton_extraction_config['flux_threshold'])

# Threhsold and thin the flux to obtain the skeleton
thin_flux = fluxToSkeletonMap(dist_grad_flux, skeleton_extraction_config['flux_threshold'])
skeleton_map = subsampled_map.copy()
skeleton_map.setData(thin_flux.astype(int))
sdg_planner.updateOnDistMapUpdate(dist_map, skeleton_map)
'''