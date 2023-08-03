import rospy
from navigation_utils.paths import WaypointsPath
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path

def waypointsPathToRosPath(waypoints_path):
    path_msg = Path()
    path_header = Header()
    path_header.frame_id = "map"
    path_header.stamp = rospy.Time(0)
    path_msg.header = path_header
    path_poses = []
    for k, path_pos in enumerate(waypoints_path.waypoints):
        pos_header = Header()
        pos_header.frame_id = "map"
        pos_header.stamp = rospy.Time(0)
        pos_stamped = PoseStamped()
        pos_stamped.header = pos_header
        pos_stamped.pose = Pose()
        pos_stamped.pose.position.x = path_pos[0]
        pos_stamped.pose.position.y = path_pos[1]
        pos_stamped.pose.orientation.w = 1
        path_poses.append(pos_stamped)
    path_msg.poses = path_poses
    return path_msg

def rosPathToWaypointsPath(path_msg):
    waypoints = []
    for p in path_msg.poses:
        pose = p.pose
        wp = [pose.position.x, pose.position.y]
        waypoints.append(wp)
    return WaypointsPath(waypoints)