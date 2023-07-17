import numpy as np
from extended_mapping.geom_processing import *

class WaypointsPath:
    """Class representing a 2D path by a list of waypoints"""

    def __init__(self, waypoints):
        "Initialization"
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

    def getSubdivized(self, nSubdivisions):
        subdWaypoints = []
        for k in range(nSubdivisions):
            t = 1.0*k/(nSubdivisions-1)
            subdWaypoints.append(self.getPosAt(t))
        subdPath = WaypointsPath(np.array(subdWaypoints))
        return subdPath

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