import numpy as np
from extended_mapping.geom_processing import *
from navigation_utils.paths import WaypointsPath
from matplotlib.collections import LineCollection
from sdg_roadmap.sdg_base import *


class BubblesPath(WaypointsPath):
    """MISSING SOME PROPER CUSTOM METHODS TO ADAPT BUBBLES RAD. WHEN WAYPOINTS ARE MODIFIED"""
    def __init__(self, bubbles_pos, radii):
        WaypointsPath.__init__(self, bubbles_pos)
        self.radii = radii

    def getAsWaypointsPath(self):
        return WaypointsPath(self.waypoints)

    @staticmethod
    def initFromWaypointsPath(waypoints_path, dist_map):
        radii = [dist_map.valueAt(p) for p in waypoints_path.waypoints]
        return BubblesPath.__init__(waypoints_path, radii)


    def getSubdivized(self, n_subdivisions, dist_map):
        """Returns the same path subdivized in the desired number of sudivisions"""
        subd_waypoints = []
        for k in range(n_subdivisions):
            t = 1.0*k/(n_subdivisions-1)
            subd_waypoints.append(self.getPosAt(t))
        new_radii = [dist_map.valueAt(p) for p in subd_waypoints]
        subdPath = BubblesPath(subd_waypoints, new_radii)
        return subdPath

    def getReducedBubbles(self): # , d_offset=0.):
        def _getNextValidInd(b0_ind, min_rad):
            next_valid_ind = b0_ind
            valid_cond = True
            while (next_valid_ind < len(self.waypoints) - 1) and valid_cond:
                valid_overlap_cond = self.radii[next_valid_ind]+self.radii[b0_ind] > np.linalg.norm(self.waypoints[next_valid_ind] - self.waypoints[b0_ind])
                valid_rad_cond = self.radii[next_valid_ind] > min_rad
                valid_cond = valid_overlap_cond #and valid_rad_cond
                if valid_cond:
                    next_valid_ind += 1
                else:
                    next_valid_ind -= 1
            return next_valid_ind
        
        if len(self.waypoints) == 2:
            return self
        min_rad = np.min(self.radii)
        final_wp = []
        final_rad = []
        prev_curr_id = 0
        curr_id = 0
        next_id = 1
        while(next_id > prev_curr_id and curr_id < len(self.waypoints)):
            final_wp.append(self.waypoints[curr_id])
            final_rad.append(self.radii[curr_id])
            next_id = _getNextValidInd(curr_id, min_rad)
            prev_curr_id = curr_id
            curr_id = next_id
        return BubblesPath(final_wp, final_rad)

    def computeSplineControlPoints(self):
        m_list = []
        for k, path_vertex in enumerate(self.waypoints[:-1]) :
            t_b = min(1, self.radii[k]/np.linalg.norm(path_vertex - self.waypoints[k+1]))
            t_bnext = min(1,self.radii[k+1]/np.linalg.norm(path_vertex - self.waypoints[k+1]))
            m = path_vertex + (t_b - 0.5*(t_b + t_bnext - 1) )*(self.waypoints[k+1] - path_vertex)
            m_list.append(m)
        
        control_points = []
        b0 = self.waypoints[0]
        b0_control_points = [
            b0 - (m_list[0] - b0)/3,
            b0,
            b0 + (m_list[0] - b0)/3,
            b0 + (m_list[0] - b0)*2/3
        ]
        control_points.extend(b0_control_points)
        
        for k,path_vertex in enumerate(self.waypoints[1:-1]) :
            b = path_vertex
            b_ind = k+1
            b_control_points = [
                b + (m_list[b_ind - 1] - b)*2/3,
                b + (m_list[b_ind - 1] - b)/3,
                b + (m_list[b_ind - 1] - b)/8 + (m_list[b_ind] - b)/8,
                b + (m_list[b_ind] - b)/3,
                b + (m_list[b_ind] - b)*2/3
            ]
            control_points.extend(b_control_points)
        
        blast = self.waypoints[-1]   
        blast_control_points = [
            blast + (m_list[-1] - blast)*2/3,
            blast + (m_list[-1] - blast)/3,
            blast,
            blast - (m_list[-1] - blast)/3,
        ]
        control_points.extend(blast_control_points)
        return control_points

    def getSmoothedSpline(self, dist_map, n_interp=10):
        control_points = self.computeSplineControlPoints()
        spline_vals = []
        for k, cp in enumerate(control_points) :
            if k==0 or k>len(control_points)-3 :
                continue
            if k == len(control_points) - 3:
                s = np.arange(n_interp)/(n_interp-1)
            else:
                s = np.arange(n_interp)/(n_interp)
            s = s.reshape(-1,1)
            loc_spline_vals = (1./6)*(
                    np.multiply((1-s)*(1-s)*(1-s),control_points[k-1]) 
                    + (3*s*s*s - 6*s*s + 4)*control_points[k]
                    + (-3*s*s*s + 3*s*s + 3*s + 1)*control_points[k+1]
                    + s*s*s*control_points[k+2])
            spline_vals.extend(loc_spline_vals)
            
        new_radii = [dist_map.valueAt(p) for p in spline_vals]
        return BubblesPath(spline_vals, new_radii)

    def display(self, show_bubbles=True, cmap=None, color="green", blw=1, label=None):
        lcol = []
        colors = []
        for k,bpos in enumerate(self.waypoints):
            if cmap is not None:
                color = cmap(k/(len(self.waypoints) - 1))
            colors.append(color)
            if show_bubbles:
                circ = plt.Circle(bpos, self.radii[k], ec=color, fc=(0,0,0,0), lw=blw)
                plt.gca().add_patch(circ)
            if k < len(self.waypoints) - 1:
                lcol.append(np.array([bpos, self.waypoints[k+1]]))
        if cmap is not None:
            colors = [cmap(i/(len(lcol) - 1)) for i in range(len(lcol))]
        plt.gca().add_collection(LineCollection(lcol, colors=colors, label=label))