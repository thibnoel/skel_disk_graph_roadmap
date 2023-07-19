import numpy as np

# Ref : Claude Samson in Nonlinear Control for Mobile robots
# We reuse the same notations
# Some variations are introduced (linear velocity limitation based on angular velocity)

def computeAngleDiff2D(ref_dir, other_dir):
    norm_prod = np.linalg.norm(ref_dir)*np.linalg.norm(other_dir)
    if norm_prod > 1e-6 :
        angle = np.arccos(np.dot(ref_dir,other_dir)/norm_prod)
    else:
        angle = 0
    if(np.cross(ref_dir, other_dir) < 0):
        angle = -angle
    return angle

class pathFollowingController:
    def __init__(self, des_lin_vel, max_ang_vel, k2, k3):
        self.des_lin_vel = des_lin_vel
        self.max_ang_vel = max_ang_vel
        self.k2 = k2
        self.k3 = k3

    def computeLocalAngVelCommand(self, angle_diff, signed_dist_to_path, local_path_curvature):
        if angle_diff < 1e-6:
            u = -self.k3*np.abs(self.des_lin_vel)*angle_diff - self.k2*self.des_lin_vel*signed_dist_to_path
        else:
            u = -self.k3*np.abs(self.des_lin_vel)*angle_diff - self.k2 * \
                self.des_lin_vel*signed_dist_to_path*np.sin(angle_diff)/angle_diff

        w = u + self.des_lin_vel*np.cos(angle_diff)*local_path_curvature/(1-signed_dist_to_path*local_path_curvature)
        return w

    def computeRawCommand(self, path, agent_position, agent_orientation):
        path_witness_point, wp_t = path.getNearestPoint(agent_position)
        wp_index = path.getWpIndexAt(wp_t)
        curvature = path.getCurvatureAt(wp_t)

        tangent = path.waypoints[wp_index+1] - path_witness_point
        normal = np.array([-tangent[1], tangent[0]])
        if self.des_lin_vel < 0:
            tangent = -tangent
            normal = -normal

        angle_diff = computeAngleDiff2D(tangent, np.array([np.cos(agent_orientation), np.sin(agent_orientation)]))
        signed_dist = np.linalg.norm(agent_position - path_witness_point)
        if np.dot(agent_position - path_witness_point, normal) < 0:
            signed_dist = -signed_dist

        w0 = self.computeLocalAngVelCommand(angle_diff, signed_dist, curvature)
        if np.abs(w0) > self.max_ang_vel:
            w0 = np.sign(w0)*self.max_ang_vel
        return [self.des_lin_vel, w0]

    def computeAdjustedCommand(self, raw_command, dist_to_obst, robot_rad, safety_factor=0.5, ang_vel_trigger_ratio=0.9):
        max_turn_radius = max(0, safety_factor*dist_to_obst - robot_rad)
        updated_lin_vel = raw_command[0]
        if np.abs(raw_command[1]) > self.max_ang_vel*ang_vel_trigger_ratio:
            max_local_lin_vel = max_turn_radius*raw_command[1]
            updated_lin_vel = np.abs(max_local_lin_vel)
        return [updated_lin_vel, raw_command[1]]

