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

### Geometric distance computations in 2D

def pointSegDistWp(point, segment, toLine=False):
    """
    Returns the closest point on a segment wrt a given pos
    """
    dp = segment[1] - segment[0]
    seg_sqr_len = np.matmul(dp.T, dp) 
    s = (dp[0]/seg_sqr_len)*(point[0] - segment[0][0]) + (dp[1]/seg_sqr_len)*(point[1] - segment[0][1])
    
    if not toLine :
        if s < 0:
            wp = segment[0]
        elif s > 1:
            wp = segment[1]
        else:
            wp = segment[0] + s*dp 
    else:
        wp = segment[0] + s*dp 
    
    return wp 



def segSegDistWp(seg1, seg2):
    """
    Returns the pair of closest points on 2 segments wrt one another
        Segments described as 
        p1 = seg1[0] + s*u
        p2 = seg2[0] + t*v
    """
    u = seg1[1] - seg1[0]
    v = seg2[1] - seg2[0]
    w = seg2[0] - seg1[0]

    uTu = np.matmul(u.T, u)
    vTv = np.matmul(v.T, v)
    wTu = np.matmul(w.T, u) 
    wTv = np.matmul(w.T, v) 
    uTv = np.matmul(u.T, v) 

    det = uTu*vTv - uTv*uTv

    # parallel case 
    if segmentsColinearityCheck(seg1, seg2) :
        candPair0 = [seg1[0], pointSegDistWp(seg1[0], seg2)]
        candPair1 = [seg1[1], pointSegDistWp(seg1[1], seg2)]
        candPair2 = [seg2[0], pointSegDistWp(seg2[0], seg1)]
        candPair3 = [seg2[1], pointSegDistWp(seg2[1], seg1)]

        candPairs = [candPair0, candPair1, candPair2, candPair3]
        candPairDists = [np.linalg.norm(cp[1] - cp[0]) for cp in candPairs]
        
        return candPairs[np.argmin(candPairDists)]

    # Compute optimal values of t and s
    s = (wTu*vTv - wTv*uTv)/det 
    t = (wTu*uTv - wTv*uTu)/det 

    # Clamp between 0 and 1
    s = max(0,min(1,s))
    t = max(0,min(1,t))

    # Recompute optim values 
    S = (t*uTv + wTu)/uTu
    T = (s*uTv - wTv)/vTv

    # Reclamp
    S = max(0,min(1,S))
    T = max(0,min(1,T))

    return seg1[0] + S*u, seg2[0] + T*v 