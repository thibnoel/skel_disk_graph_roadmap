#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-

import rospy
import time 

import matplotlib.pyplot as plt

from extended_mapping.map_processing import * 
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, MapMetaData
from nav_msgs.srv import GetMap, GetMapResponse


# Separate the occupancy preprocessing currently situated in the distance_map_server
class OccupancyPreprocessingServer:
    """ROS node responsible for preprocessing of the occupancy map"""
    def __init__(self, occ_map_service, subsampling, obstacles_threshold, dilation_erosion_dist):
        # Parameters
        self.subsampling = subsampling
        self.obstacles_threshold = obstacles_threshold
        self.dilation_erosion_dist = dilation_erosion_dist

        # Service proxies
        self.occ_map_service = rospy.ServiceProxy(occ_map_service, GetMap)

        # Publishers
        self.processed_map_publisher = rospy.Publisher("~processed_map", OccupancyGrid, queue_size=1, latch=True)
        self.processed_binary_map_publisher = rospy.Publisher("~processed_binary_map", OccupancyGrid, queue_size=1, latch=True)

        # Service
        self.processed_binary_map_service = rospy.Service("~get_processed_binary_map", GetMap, self.processedMapBinaryServiceCallback)
        self.processed_map_service = rospy.Service("~get_processed_map", GetMap, self.processedMapServiceCallback)

        # State
        self.input_map = None
        self.processed_map = None

    def processMap(self, raw_map_msg):
        #raw_map_msg = self.occ_map_service().map
        if self.input_map is None:
            self.input_map = EnvironmentMap.initFromRosOccMsg(raw_map_msg)
        occ_data = ROSOccMapToArray(raw_map_msg)

        self.input_map.setResolution(raw_map_msg.info.resolution)
        self.input_map.setDim([raw_map_msg.info.width, raw_map_msg.info.height])
        self.input_map.setOrigin([raw_map_msg.info.origin.position.x, raw_map_msg.info.origin.position.y])
        self.input_map.setData(occ_data)

        rospy.loginfo("Occupancy map received")
        
        # Subsampling
        subsampled_map = self.input_map.copy()
        if self.subsampling != 1:
            subsampled_map = subsampleMap(self.input_map, self.subsampling)
        # Extracting obstacles
        obst_map = subsampled_map.copy()
        obst_map.setData(subsampled_map.data > self.obstacles_threshold)
        # Dilation + erosion
        filtered_map = mapDilateErode(obst_map, self.dilation_erosion_dist)
        self.processed_map = subsampled_map.copy()
        #self.processed_map.data[np.where(subsampled_map.data == 0.5)] = -1/100 # Set unknown cells as ROS expects
        self.processed_map.data[np.where(filtered_map.data == 1)] = 1
        

        rospy.loginfo("Preprocessing finished")


    def processedMapServiceCallback(self, req):
        """Returns the processed map"""
        raw_map_msg = self.occ_map_service().map
        self.processMap(raw_map_msg)
        response = GetMapResponse()
        response.map = OccupancyGrid()
        response.map.header = raw_map_msg.header
        response.map.info.resolution = self.processed_map.resolution
        response.map.info.origin.position = Point(self.processed_map.origin[0], self.processed_map.origin[1], raw_map_msg.info.origin.position.z)
        response.map.info.width = self.processed_map.dim[0]
        response.map.info.height = self.processed_map.dim[1]
        response.map.data = (100*self.processed_map.data.T).astype(np.int32).flatten()

        self.processed_map_publisher.publish(response.map)
        return response

    def processedMapBinaryServiceCallback(self, req):
        """Reurns the processed map as a binary obstacles map"""
        raw_map_msg = self.occ_map_service().map
        self.processMap(raw_map_msg)
        modified_data = self.processed_map.data.copy()
        modified_data[np.where(modified_data < 0)] = 0

        response = GetMapResponse()
        response.map = OccupancyGrid()
        response.map.header = raw_map_msg.header
        response.map.info.resolution = self.processed_map.resolution
        response.map.info.origin.position = Point(self.processed_map.origin[0], self.processed_map.origin[1], raw_map_msg.info.origin.position.z)
        response.map.info.width = self.processed_map.dim[0]
        response.map.info.height = self.processed_map.dim[1]
        response.map.data = (100*modified_data.T).astype(np.int32).flatten()

        self.processed_binary_map_publisher.publish(response.map)
        return response


if __name__=='__main__':
    rospy.init_node("occ_preproc_server")
    r = rospy.Rate(10)

    time.sleep(5)
    
    # Parameters
    occupancy_map_service = rospy.get_param("~occupancy_map_service", "rtabmap/get_map")
    obstacles_threshold = rospy.get_param("~obstacles_threshold", 0.51)
    subsampling = rospy.get_param("~subsampling", 1.)
    dil_erosion_dist = rospy.get_param("~dil_erosion_dist", 1.5)

    rospy.wait_for_service(occupancy_map_service)
    occ_preproc_server = OccupancyPreprocessingServer(occupancy_map_service, subsampling, obstacles_threshold, dil_erosion_dist)
    
    
    while not rospy.is_shutdown():
        r.sleep()
    rospy.spin()