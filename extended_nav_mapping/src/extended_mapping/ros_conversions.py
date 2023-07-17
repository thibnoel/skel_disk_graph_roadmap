import numpy as np
from extended_mapping.map_processing import EnvironmentMap

from extended_nav_mapping.msg import EnvironmentGridMap
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion
from nav_msgs.msg import MapMetaData

def ROSEnvMapToArray(env_map_msg):
    """Extracts the data from a custom EnvironmentGridMap ROS message to a numpy array"""
    info = env_map_msg.info
    data = env_map_msg.data
    data = np.array(data)
    data = data.reshape((info.height, info.width)).T
    return data


def ROSOccMapToArray(occ_map_msg):
    """Extracts the data from a nav_msgs/OccupancyGrid ROS message to a numpy array"""
    data = ROSEnvMapToArray(occ_map_msg)
    data[np.where(data == -1)] = 50
    return 0.01*data


def envMapToROSMsg(env_map, frame_id="map"):
    """Converts an EnvironmentMap to a custom EnvironmentGridMap ROS message"""
    env_map_msg = EnvironmentGridMap()
    env_map_msg.header = Header()
    env_map_msg.header.frame_id = frame_id
    env_map_msg.info.resolution = env_map.resolution
    env_map_msg.info.width = env_map.dim[0]
    env_map_msg.info.height = env_map.dim[1]
    env_map_msg.info.origin.position = Vector3(env_map.origin[0], env_map.origin[1],0)
    env_map_msg.info.origin.orientation = Quaternion(0,0,0,1)
    env_map_msg.data = (env_map.data.T).flatten().astype(float)
    return env_map_msg

def envMapFromRosOccMsg(env_map_msg):
    """Initializes an EnvironmentMap from a ROS OccupancyGrid message"""
    env_map = EnvironmentMap(
            env_map_msg.info.resolution,
            [env_map_msg.info.width, env_map_msg.info.height],
            [env_map_msg.info.origin.position.x, env_map_msg.info.origin.position.y]
        )
    env_map.setData(ROSOccMapToArray(env_map_msg))
    return env_map

def envMapFromRosEnvGridMapMsg(env_map_msg):
    """Initializes an EnvironmentMap from a ROS custom EnvironmentGridMap message"""
    env_map = EnvironmentMap(
            env_map_msg.info.resolution,
            [env_map_msg.info.width, env_map_msg.info.height],
            [env_map_msg.info.origin.position.x, env_map_msg.info.origin.position.y]
        )
    env_map.setData(ROSEnvMapToArray(env_map_msg))
    return env_map