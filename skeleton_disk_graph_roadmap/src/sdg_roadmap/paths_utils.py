import numpy as np
from extended_mapping.geom_processing import *
from matplotlib.collections import LineCollection
from sdg_roadmap.sdg_base import *

class WaypointsPath:
    """Class representing a 2D path by a list of waypoints"""

    def __init__(self, waypoints):
        """Initialization"""
        self.waypoints = np.array(waypoints)
        # Compute the sub paths lengths
        curr_waypoints = self.waypoints[:-1, :]
        waypoints_offset = self.waypoints[1:, :].copy()
        diff = waypoints_offset - curr_waypoints
        self.lengths = np.linalg.norm(diff, axis=1)

    def filterOnDist(self, filterDist):
        filtered_wp = [self.waypoints[0]]
        for k, wp in enumerate(self.waypoints[1:]):
            if np.linalg.norm(wp - self.waypoints[k-1]) < filterDist:
                continue
            filtered_wp.append(wp)
        self = WaypointsPath(np.array(filtered_wp))

    def rescale(self, scale):
        for p in self.waypoints:
            p[0] = scale*p[0]
            p[1] = scale*p[1]
        return WaypointsPath(self.waypoints)

    def getTotalLength(self):
        """Returns the total length of the path"""
        return np.sum(self.lengths)

    def getWpIndexAt(self, t):
        cumul = np.sum(self.lengths)
        t_length = t*cumul
        curr_length = 0

        wp_index = 0
        for k in range(len(self.lengths)):
            curr_length += self.lengths[k]
            if t_length <= curr_length:
                wp_index = k
                break
        return wp_index

    def getPosAt(self, t):
        """Gets the pos t, linearly interpolated along the path between the start (t=0) and end (t=1)"""
        if t <= 0:
            return self.waypoints[0]
        if t >= 1:
            return self.waypoints[-1]
        cumul = np.sum(self.lengths)
        t_length = t*cumul
        curr_length = 0

        wp_index = 0
        t_res = 1
        for k in range(len(self.lengths)):
            curr_length += self.lengths[k]
            if t_length <= curr_length:
                wp_index = k
                t_res = 1 - (curr_length - t_length)/self.lengths[k]
                break
        return self.waypoints[wp_index] + t_res*(self.waypoints[wp_index+1] - self.waypoints[wp_index])

    def getCurvatureAtInd(self, ind):
        if ind == 0 or ind == len(self.waypoints) - 1:
            return 0
        else:
            p1 = self.waypoints[ind-1]
            p2 = self.waypoints[ind]
            p3 = self.waypoints[ind+1]
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)

            b = b/a
            c = c/a
            a = 1

            if np.abs((a+b) - c) < 1e-6:
                return 0
            q = (a*a + b*b - c*c)/(2*a*b)
            r = c/(2*np.sqrt(1 - q*q))
            return 1/r

    def getCurvatureAt(self, t):
        #tot_length = self.getTotalLength()
        ind = self.getWpIndexAt(t)
        return self.getCurvatureAtInd(ind)

    def getNearestPoint(self, pos):
        """Computes the closest point on the path to a given pos"""
        segments = [[self.waypoints[i], self.waypoints[i+1]]
                    for i in range(len(self.waypoints) - 1)]
        global_min = float('inf')
        global_min_ind = -1
        closest_point = None
        for k, seg in enumerate(segments):
            s_wp = pointSegDistWp(pos, seg)
            dist = np.linalg.norm(pos - s_wp)
            if dist < global_min:
                global_min = dist
                global_min_ind = k
                closest_point = s_wp
            #print(k, dist)
        t_on_path = (np.sum(self.lengths[:global_min_ind]) + np.linalg.norm(
            closest_point - segments[global_min_ind][0]))/self.getTotalLength()
        return closest_point, t_on_path

    def getSubdivized(self, n_subdivisions):
        """Returns the same path subdivized in the desired number of sudivisions"""
        subd_waypoints = []
        for k in range(n_subdivisions):
            t = 1.0*k/(n_subdivisions-1)
            subd_waypoints.append(self.getPosAt(t))
        subdPath = WaypointsPath(np.array(subd_waypoints))
        return subdPath

    '''
    def simplifyToMinimalLines(self, costMap, mapOrigin, mapRes, costThresh, show=False):
        def maxCostBetween(startPos, endPos, lineRes):
            # Returns the max cost encountered between parent and node
            max_cost = -float('inf')
            length = np.linalg.norm(endPos - startPos)
            nPos = int(length/lineRes)
            posList = [startPos + (1.0*i/(nPos-1))*(endPos - startPos)
                       for i in range(nPos)] + [endPos]
            for pos in posList:
                pix_pos = worldPos_to_pixPos(np.array(pos), mapOrigin, mapRes)
                max_cost = max(max_cost, costMap[pix_pos[0], pix_pos[1]])
            return max_cost

        LINE_RES = 0.005

        new_waypoints = [self.waypoints[0]]
        solvedUpTo = 0
        curr_try_ind = len(self.waypoints) - 1
        curr_cost = maxCostBetween(
            self.waypoints[solvedUpTo], self.waypoints[curr_try_ind], LINE_RES)
        lines_count = 0
        while(solvedUpTo < len(self.waypoints)-1):
            while(curr_cost > costThresh and curr_try_ind > solvedUpTo + 1):

                if show:
                    plot_points = np.vstack(
                        (self.waypoints[solvedUpTo], self.waypoints[curr_try_ind]))
                    plt.plot(plot_points[:, 0],
                             plot_points[:, 1], color='red', lw=0.5)

                curr_try_ind -= 1
                curr_cost = maxCostBetween(
                    self.waypoints[solvedUpTo], self.waypoints[curr_try_ind], LINE_RES)

            if show:
                plot_points = np.vstack(
                    (self.waypoints[solvedUpTo], self.waypoints[curr_try_ind]))
                plt.plot(plot_points[:, 0], plot_points[:, 1],
                         color='green', lw=2, marker='.')

            new_waypoints.append(self.waypoints[curr_try_ind])
            solvedUpTo = curr_try_ind

            curr_try_ind = len(self.waypoints) - 1
            curr_cost = maxCostBetween(
                self.waypoints[solvedUpTo], self.waypoints[curr_try_ind], LINE_RES)
            lines_count += 1

        print("Path simplification solved - {} lines remaining".format(lines_count))
        return WaypointsPath(np.array(new_waypoints))
    '''

    def computeDthetaList(self, init_theta=0):
        # Modif dtheta_list indexing to match redaction
        #dtheta_list = []
        dtheta_list = [init_theta]
        theta = init_theta
        for k in range(len(self.waypoints) - 1):
            p = self.waypoints[k]
            next_p = self.waypoints[k+1]
            dp = next_p - p
            new_theta = np.arctan2(dp[1], dp[0])
            dtheta = new_theta - theta
            # if k > 0:
            dtheta_list.append(dtheta)
            #theta += dtheta
            theta = new_theta
        return dtheta_list

    def show(self, color='blue', marker=',', linewidth=1, label=None):
        #t = 0
        #dt = 1/n_points
        for k in range(len(self.waypoints) - 1):
            #p0 = self.getPosAt(t)
            #p1 = self.getPosAt(t+dt)
            p0 = self.waypoints[k]
            p1 = self.waypoints[k+1]
            if k == 0:
                plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, marker=marker,
                         linewidth=linewidth, markersize=3*linewidth, label=label)
            else:
                plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color,
                         marker=marker, linewidth=linewidth, markersize=3*linewidth)
            #t = t+dt


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