import numpy as np
from extended_mapping.map_processing import *
from skimage.morphology import thin


def getNeihgborsCoord(pix_coord, include_diag=True):
    # Order : right, up, left, down
    i,j = pix_coord
    neighbors = [
        [i+1, j],
        [i  , j+1],
        [i-1, j],
        [i  , j-1]
    ]
    diag_neighbors = [
        [i+1, j+1],
        [i-1, j+1],
        [i-1, j-1],
        [i+1, j-1]
    ]
    if not include_diag :
        return neighbors
    all_neighbors = []
    for k in range(4):
        all_neighbors.append(neighbors[k])
        all_neighbors.append(diag_neighbors[k])
    return all_neighbors


def gradFlux(grad_q, include_diag_neighbors=True):
    phi = np.zeros(grad_q.shape[1:])

    grad = grad_q.transpose(1,0,2)

    xshift_pos = np.dot(np.array([1, 0]), np.roll(grad, 1, axis=0))
    xshift_neg = np.dot(np.array([-1, 0]), np.roll(grad, -1, axis=0))
    yshift_pos = np.dot(np.array([0,1]), np.roll(grad, 1, axis=2))
    yshift_neg = np.dot(np.array([0,-1]), np.roll(grad, -1, axis=2))

    if include_diag_neighbors :
        ur_shift = np.dot(np.array([1, 1]), np.roll(grad, (1,1), axis=(0,2)))
        ul_shift = np.dot(np.array([-1, 1]), np.roll(grad, (-1,1), axis=(0,2)))
        bl_shift = np.dot(np.array([-1, -1]), np.roll(grad, (-1,-1), axis=(0,2)))
        br_shift = np.dot(np.array([1, -1]), np.roll(grad, (1,-1), axis=(0,2)))

        phi = 0.5*np.sqrt(2)*(ur_shift + ul_shift + bl_shift + br_shift)

    phi = phi + xshift_pos + xshift_neg + yshift_pos + yshift_neg
    phi[0,:] = 0
    phi[-1,:] = 0
    phi[:,0] = 0
    phi[:,-1] = 0

    return -phi/(2*np.pi)


def fluxToSkeletonMap(flux_map, flux_threshold):
    """Wraps scikit image thin() to return a thin skeleton from a distance gradient flux map"""
    masked_flux = flux_map.copy()
    masked_flux[np.where(masked_flux > flux_threshold)] = 0
    return thin(masked_flux < 0)
