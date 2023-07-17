"""
Generic class definition to manage a 2D environment map

General idea : 
    - handle map-to-world / world-to-map coordinates
    - modular input/outputs
    - build a pipeline of such processors to extract useful info from the map while remaining modular
    - basis for a ROS node providing the in/out as service
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from scipy.interpolate import interp2d
from scipy.ndimage import distance_transform_edt
from extended_mapping.geom_processing import distancesPointToSet

def imageToArray(img_path, transpose=False):
    """Reads an image file to a numpy array"""
    img = cv2.imread(img_path, 1)
    img_array = 1 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
    if transpose:
        img_array = img_array.T
    return img_array


class EnvironmentMap:
    """Data class for a generic 2D scalar map prepresenting the environment"""    
    def __init__(self, resolution, dim, origin, data=None):
        """
        Initialization method

        Arguments :
            resolution (float): the length of a pixel in meters, assuming square pixels
            dim ([int,int]): the dimensions ([width, height]) of the 2D map
            origin (np.array([float, float])): the 2D coordinates of the map origin in the world frame
            data (np.array): the map data as an array of shape dim
        """
        self.resolution = resolution 
        self.dim = dim
        self.origin = origin
        self.data = None
        #self.interpolator = None
        if data is None :
            self.setData(np.zeros(self.dim))
        else:
            self.setData(data)

    def copy(self):
        """Returns a copy of this EnvironmentMap"""
        copy = EnvironmentMap(self.resolution, self.dim, self.origin, data=self.data.copy())
        return copy

    def setResolution(self, res):
        """Sets the map resolution"""
        self.resolution = res

    def setDim(self, dim):
        """Sets the map dimensions"""
        self.dim = dim
        self.data = np.zeros(self.dim)

    def setData(self, data):
        """Sets the data array"""
        if not (np.array(data.shape) == self.dim).all :
            print("INCORRECT DATA SHAPE")
            return
        self.data = data

    def setOrigin(self, origin):
        """Sets the map origin"""
        self.origin = origin

    def mapToWorldCoord(self, map_pos):
        """Converts from map pixel coordinates to world frame coordinates"""
        scaled_pos = (np.array(map_pos) + np.array([0.5,0.5]))*self.resolution 
        return scaled_pos + self.origin

    def worldToMapCoord(self, world_pos):
        """Converts from world frame coordinates to map pixel coordinates"""
        transl_pos = np.array(world_pos) - self.origin
        map_pix_pos = (np.floor(transl_pos/self.resolution - np.array([0.5,0.5]))).astype(int)
        return map_pix_pos

    def getNpMeshgrid(self):
        """Returns the numpy meshgrid associated to the map, i.e. a set of the discretized world positions it contains"""
        x = self.origin[0] + self.resolution*np.arange(0, self.dim[0])
        y = self.origin[1] + self.resolution*np.arange(0, self.dim[1])
        X, Y = np.meshgrid(x, y)
        return X,Y

    def isInBounds(self, world_pos):
        """Checks if a given world pos. is contained within the map boundaries"""
        pix_pos = self.worldToMapCoord(world_pos)
        if (
            (pix_pos[0] < 0) or 
            (pix_pos[0] > self.dim[0] - 1) or 
            (pix_pos[1] < 0) or 
            (pix_pos[1] > self.dim[1] - 1) 
        ):
            return False
        return True

    def valueAt(self, world_pos):
        """Gets the map data at a given world pos, by discretizing it"""
        if self.isInBounds(world_pos):
            pix_pos = self.worldToMapCoord(world_pos)
            return self.data[pix_pos[0], pix_pos[1]]
        return float("nan")

    def interpolateAt(self, world_pos_x, world_pos_y):
        """Gets the map data at a given world pos, by interpolating it"""
        interpolator = interp2d(self.origin[0] + self.resolution*np.array(range(self.dim[1])), self.origin[1] + self.resolution*np.array(range(self.dim[0])), self.data)
        return interpolator(world_pos_x, world_pos_y)

    def getExtent(self, transpose=False):
        """Returns the correct axis extent for visualization of the map using matplotlib"""
        extent = [
            self.origin[0], 
            self.origin[0] + self.dim[0] * self.resolution,
            self.origin[1],
            self.origin[1] + self.dim[1] * self.resolution
        ]
        if transpose:
            extent = [extent[3], extent[2], extent[0], extent[1]]
        return extent

    def display(self, cmap=plt.cm.viridis):
        """Visualizes the map using matplotlib"""
        plt.imshow(np.flip(self.data.T, axis=0), extent=self.getExtent(transpose=False), cmap=cmap)


def subsampleMap(source_map, subsampling_factor):
    subsamp_dim = (np.array(source_map.dim)/subsampling_factor).astype(int)
    subsamp_res = (source_map.resolution*source_map.dim[0])/subsamp_dim[0]
    target_map = EnvironmentMap(subsamp_res, subsamp_dim, source_map.origin)
    new_data = source_map.interpolateAt(source_map.origin[0] + subsamp_res*np.array(range(subsamp_dim[1])), 
        source_map.origin[1] + subsamp_res*np.array(range(subsamp_dim[0])))
    target_map.setData(new_data)
    return target_map


def extractSubMap(source_map, bb_topleft, bb_bottomright):    
    pix_top_left = source_map.worldToMapCoord(bb_topleft)#target_map.origin + np.multiply(target_map.resolution*np.array(target_map.dim), [0,1]) )
    print(pix_top_left)
    pix_bottom_right = source_map.worldToMapCoord(bb_bottomright)#target_map.origin + np.multiply(target_map.resolution*np.array(target_map.dim), [1,0]))
    print(pix_bottom_right)

    pix_top_left[0] = max(pix_top_left[0], 0)
    pix_top_left[1] = min(pix_top_left[1], source_map.dim[1]-1)
    pix_bottom_right[0] = min(pix_bottom_right[0], source_map.dim[0]-1)
    pix_bottom_right[1] = max(pix_bottom_right[1], 0)

    data = source_map.data[pix_top_left[0]:pix_bottom_right[0], pix_bottom_right[1]:pix_top_left[1]]

    target_dim = data.shape
    target_origin = source_map.mapToWorldCoord(pix_top_left) #bb_topleft + np.array([0, -target_dim[1]*source_map.resolution])
    target_map = EnvironmentMap(source_map.resolution, target_dim, target_origin)
    target_map.setData(data)
    #target_map.display()
    #plt.show()
    return target_map


def computeDistsScipy(source_map, obst_min_threshold, obst_d_offset=0, compute_negative_dist=False):
    """
    Computes the signed distance filed (and its gradient) associated to an EnvironmentMap representing occupancy
    
    Arguments:
        - source_map: occupancy map as an EnvironmentMap with values in [0,1]
        - obst_min_threshold: occupancy threhsold used to binarize the map
        - obst_d_offset: offset distance added to the distance field
        - compute_negative_dist: if set to True, the distance is set to 0 inside obstacles, making the computation a bit cheaper

    Returns:
        - distance map: EnvironmentMap containing the distance values
        - [dist_xgrad_map, dist_ygrad_map]: pair of EnvironmentMap containing the x and y components of the distance field gradient
    """
    # Initialize output maps
    distance_map = EnvironmentMap(source_map.resolution, source_map.dim, source_map.origin)
    dist_xgrad_map = EnvironmentMap(source_map.resolution, source_map.dim, source_map.origin)
    dist_ygrad_map = EnvironmentMap(source_map.resolution, source_map.dim, source_map.origin)
    
    # Positive distance field
    d = distance_transform_edt((1-(source_map.data > obst_min_threshold)), return_indices=True)
    dist_field = source_map.resolution*d[0]
    wp = source_map.resolution*d[1]

    # Negative distance field
    if compute_negative_dist:
        neg_d = distance_transform_edt((1-(source_map.data < obst_min_threshold)), return_indices=True)
        dist_field = dist_field - source_map.resolution*neg_d[0]
        wp[0][np.where(dist_field < 0)] = (source_map.resolution*neg_d[1])[0][np.where(dist_field < 0)]
        wp[1][np.where(dist_field < 0)] = (source_map.resolution*neg_d[1])[1][np.where(dist_field < 0)]

    # Compute gradients maps
    # Initialize map grid positions
    X,Y = source_map.resolution*np.array(np.meshgrid(range(source_map.dim[1]), range(source_map.dim[0])))
    pos_array = np.array([Y,X])
    grad_x = (pos_array[0,:,:] - wp[0,:,:])/dist_field
    grad_y = (pos_array[1,:,:] - wp[1,:,:])/dist_field
    grad_x[np.isnan(grad_x)] = 0
    grad_y[np.isnan(grad_y)] = 0
    # Set output maps data
    if compute_negative_dist:
        dist_mask = np.ones(dist_field.shape)
    else:
        dist_mask = dist_field > obst_d_offset
    distance_map.setData((dist_field - obst_d_offset)*dist_mask)
    dist_xgrad_map.setData(grad_x*dist_mask)
    dist_ygrad_map.setData(grad_y*dist_mask)
    return distance_map, [dist_xgrad_map, dist_ygrad_map]

def mapDilateErode(binary_map, dist):
    """
    Applies dilation and erosion (in this order) operations with the same kernel size to a binary map
    
    Arguments:
        - binary_map: an EnvironmentMap containing binary data
        - dist: the distance, in meters, used to define the kernel size for erosiion and dilation

    Returns:
        A dilated + eroded copy of the input map
    """
    kernel_size = int(dist/binary_map.resolution)
    kernel_size += (1 - (kernel_size % 2))
    #print("Erosion dist. : {}\nKernel size : {}".format(erd_dist, erd_kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #walls_diff_eroded = cv2.erode((binary_map.data > 0).astype(float), erd_kernel, iterations=1)
    map_dilated = cv2.dilate((binary_map.data > 0).astype(float), kernel, iterations=1)
    map_eroded = cv2.erode(map_dilated, kernel, iterations=1)
    
    filtered_map = binary_map.copy()
    filtered_map.setData(map_eroded)
    return filtered_map