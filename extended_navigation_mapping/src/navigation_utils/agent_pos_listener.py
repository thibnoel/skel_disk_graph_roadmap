import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion
import numpy as np


class AgentPosListener:
    def __init__(self, map_frame, agent_frame):
        self.map_frame = map_frame
        self.agent_frame = agent_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get2DAgentPos(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.agent_frame, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return [0, 0]

        agent_pos = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y
        ])
        return agent_pos

    def get3DAgentPos(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.agent_frame, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return [0, 0, 0]

        agent_pos = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ])
        return agent_pos

    def get2DAgentPosRot(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.agent_frame, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return [0, 0, 0]

        agent_rot = euler_from_quaternion(
            np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
        )

        agent_pos_rot = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            # + 0.5*np.pi # rotate 90deg to account for badly-oriented base_link in fake_pepper model
            agent_rot[2]
        ])
        return agent_pos_rot

    # Compute the distance and angle between 2 nodes in world coordinates
    @staticmethod
    def computeDistAndAngle(from_pos, to_pos):
        diff = to_pos - from_pos
        d = np.hypot(diff[0], diff[1])
        theta = np.arctan2(diff[1], diff[0])
        return d, theta