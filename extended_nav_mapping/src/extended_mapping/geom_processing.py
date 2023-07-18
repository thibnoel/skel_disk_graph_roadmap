import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist
from skimage.draw import disk

def distancesPointToSet(point, points_set):
    """
    Computes the distances between a point and a set of points.

    Args:
        - point: 1-dimensionnal numpy array of shape (d,), where d is the dimension of the space in which the considered point is
        - points_set: numpy array of shape (n, d)
    
    Returns:
        A numpy array of shape (n,) containing the distances from point to each point in points_set 
    """
    if len(points_set) == 0:
        return float('inf')
    ref_point = point.reshape(1,-1)
    return cdist(ref_point, points_set)[0]

def circularMask(mask_dim, center, radius):
    """Creates a 2D circular mask (True inside the defined circle) of shape env_dim. The center and radius are specified in pixel units"""
    mask = np.zeros(mask_dim, dtype=bool)
    rr, cc = disk(center, radius, shape=mask_dim)
    mask[rr, cc] = 1
    return mask
