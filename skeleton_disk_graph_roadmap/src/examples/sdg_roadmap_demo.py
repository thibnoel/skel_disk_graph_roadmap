import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize

from extended_mapping.map_processing import *
#from extended_mapping.farthest_point_sampling import *
from sdg_roadmap.skel_disk_graph_provider import *

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
dist_map, grad_maps = computeDistsScipy(filtered_bin_map, map_preproc_config['obst_thresh'], obst_d_offset=map_preproc_config['dist_field_offset'], compute_negative_dist=True)
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

#skeleton_map.display()
#plt.show()

sdg_construction_config = {
    "min_jbubbles_rad" : 0.6,
    "min_pbubbles_rad" : 0.4,
    "bubbles_dist_offset" : 0.,
    "knn_checks" : 100
}

sdg_provider = SkeletonDiskGraphProvider(sdg_construction_config)
sdg_provider.updateOnDistMapUpdate(dist_map, skeleton_map)