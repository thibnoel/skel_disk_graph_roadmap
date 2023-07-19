import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize

from extended_mapping.map_processing import *
#from extended_mapping.farthest_point_sampling import *
from sdg_roadmap.skel_disk_graph_provider import *

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

if __name__ == "__main__":
    # Map file path and configuration
    maps_path_prefix = "/root/sdg_dev_catkin_ws/src/skel_disk_graph_roadmap/skeleton_disk_graph_roadmap/environments/"
    map_file_config = {
        'map_path': "env_intellab.png",
        'map_resolution': 0.1,
        'map_origin': np.array([0,0])
    }

    ### Map initialization
    map_data = imageToArray(maps_path_prefix + map_file_config['map_path'])
    env_map = EnvironmentMap(
        map_file_config['map_resolution'],
        map_data.shape,
        map_file_config['map_origin'],
        data = map_data
    )
    #env_map = extractSubMap(env_map, [0,25], [25,0])

    # Map preprocessing configuration
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
    dist_map, grad_maps = computeDistsScipy(filtered_bin_map, map_preproc_config['obst_thresh'], obst_d_offset=map_preproc_config['dist_field_offset'], compute_negative_dist=True)
    dist_grad_array = np.array([grad_maps[0].data, grad_maps[1].data])

    # Skeleton extraction - only 1 config. param, the flux threhsold
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

    SHOW_MAP_PREPROC = True
    # Display map processing steps
    if SHOW_MAP_PREPROC:
        plt.figure()
        plt.title("Source Occupancy Map")
        env_map.display(plt.cm.binary)
        plt.figure()
        plt.subplot(231)
        plt.title("Preproc. Occ.")
        filtered_bin_map.display(plt.cm.binary)
        plt.subplot(232)
        plt.title("SDF")
        distanceFigure(map_preproc_config, dist_map, plt.cm.RdYlGn, contour_color='blue', contour_lw=3, title=False)
        plt.subplot(233)
        plt.title("SDF Gradient")
        distanceGradFigure(map_preproc_config, grad_maps)
        plt.subplot(234)
        plt.title("Grad. Flux")
        grad_flux_map.display(cmap=plt.cm.binary)
        plt.subplot(235)
        plt.title("Thresh. Flux")
        thresh_flux_map.display(cmap=plt.cm.binary)
        plt.subplot(236)
        plt.title("Thin Skel.")
        skeleton_map.display(cmap=plt.cm.binary)

    sdg_construction_config = {
        "min_jbubbles_rad" : 0.6,
        "min_pbubbles_rad" : 0.4,
        "bubbles_dist_offset" : 0.,
        "knn_checks" : 100
    }

    sdg_provider = SkeletonDiskGraphProvider(sdg_construction_config)
    sdg_provider.updateOnDistMapUpdate(dist_map, skeleton_map)

    ### Path planning
    world_start = [13,10]
    world_goal = [50,46]
    path_subdiv_length = 0.5

    path = sdg_provider.getWorldPath(world_start, world_goal)
    simplified_path = path.getSubdivized(int(path.getTotalLength()/path_subdiv_length), dist_map).getReducedBubbles()
    spline_path = simplified_path.getSmoothedSpline(dist_map)

    # Path planning visualization
    SHOW_PATH_PLANNING = True
    if SHOW_PATH_PLANNING:
        plt.figure()
        plt.subplot(121)
        graph_color = 'blue'
        env_map.display(cmap=plt.cm.binary)
        sdg_provider.display(True, True, True, ecolor=graph_color, bcolor=graph_color)
        plt.title("Roadmap Graph")
        
        plt.subplot(122)
        path_cmap = None # plt.cm.viridis
        path_color = 'red'
        simplified_path_color = 'green'
        spline_path_color = 'blue'
        env_map.display(cmap=plt.cm.binary)
        path.display(show_bubbles=True, color=path_color, cmap=path_cmap, label='Raw path')
        simplified_path.display(show_bubbles=False, color=simplified_path_color, cmap=path_cmap, label='Minimal path')
        spline_path.display(show_bubbles=False, color=spline_path_color, cmap=path_cmap, label='Spline path')
        plt.legend()
        plt.title("Path Planning")
    

    ### Frontiers extraction
    source_pos = np.array([42,18])
    map_unkn_range = [0.04, 0.1]
    frontier_known_threhsold = 0.5 # ratio of known space to total surface covered by a bubble to be considered known
    search_dist_increment = 100
    path_cost_param = {
        "energy_penalty": -1,
        "coverage_reward": 0.5
    }

    exploration_provider = SDGExplorationPathProvider(sdg_provider, map_unkn_range, frontier_known_threhsold, search_dist_increment)
    frontiers_paths = exploration_provider.getFrontiersPaths(source_pos, env_map)
    best_path = exploration_provider.selectExplorationPath(frontiers_paths, path_cost_param)

    SHOW_FRONTIERS_EXTRACTIION = True
    if SHOW_FRONTIERS_EXTRACTIION:
        plt.figure()
        plt.title("Frontiers extraction")
        env_map.display(cmap=plt.cm.binary)
        length_cost_cmap = plt.cm.winter
        for path_id in frontiers_paths:
            frontiers_paths[path_id]['path'].display(show_bubbles=False, color=length_cost_cmap(frontiers_paths[path_id]['costs']['energy_penalty']))
        best_path['path'].display(show_bubbles=False, color='red', lw=3)

    plt.show()