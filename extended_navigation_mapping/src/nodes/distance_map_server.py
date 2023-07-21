#! /usr/bin/env python3
# -*- encoding: UTF-8 -*-


import rospy
import time
import numpy as np
import threading
import matplotlib.pyplot as plt

from extended_mapping.map_processing import * #ROSEnvMapToArray, ROSOccMapToArray, EnvironmentMap
from extended_mapping.ros_conversions import *
from extended_mapping.flux_skeletons_utils import gradFlux, fluxToSkeletonMap

from nav_msgs.msg import MapMetaData, OccupancyGrid
from nav_msgs.srv import GetMap
from geometry_msgs.msg import Twist, Vector3, Pose, PoseStamped, Point, Quaternion
from extended_navigation_mapping.msg import EnvironmentGridMap
from extended_navigation_mapping.srv import GetDistance, GetDistanceResponse, GetDistanceSkeleton, GetDistanceSkeletonResponse


class DistanceMapServer:
    """ROS node to provide a distance map and its gradients, by processing an occupancy map"""
    def __init__(self, occ_map_service, distance_offset, skeleton_flux_threshold):
        # Parameters
        self.distance_offset = distance_offset
        self.skeleton_flux_threshold = skeleton_flux_threshold
        self.occ_map_service = occ_map_service

        # Service proxies
        self.occ_map_service = rospy.ServiceProxy(self.occ_map_service, GetMap)
        # Publishers
        self.dist_map_publisher = rospy.Publisher("~distance", EnvironmentGridMap, queue_size=1, latch=True)
        self.dist_as_occ_publisher = rospy.Publisher("~dist_as_occ", OccupancyGrid, queue_size=1, latch=True)
        self.dist_gradX_publisher = rospy.Publisher("~grad_x", EnvironmentGridMap, queue_size=1, latch=True)
        self.dist_gradY_publisher = rospy.Publisher("~grad_y", EnvironmentGridMap, queue_size=1, latch=True)
        self.skel_map_publisher = rospy.Publisher("~skeleton", EnvironmentGridMap, queue_size=1, latch=True)
        # Distance service
        self.distance_service = rospy.Service("~get_distance", GetDistance, self.distanceServiceCallback)
        self.dist_skeleton_service = rospy.Service("~get_distance_skeleton", GetDistanceSkeleton, self.distanceSkeletonServiceCallback)
        # State
        self.input_map = None
        self.dist_map = None
        self.dist_occ_map = None
        self.dist_grad_x = None
        self.dist_grad_y = None
        self.skeleton_map = None

        self.lock = threading.Lock()

    def occMapCallback(self, occ_grid_msg):
        """Input map callback"""
        if self.input_map is None :
            self.input_map = envMapFromRosOccMsg(occ_grid_msg)
        occ_data = ROSOccMapToArray(occ_grid_msg)
        
        self.input_map.setResolution(occ_grid_msg.info.resolution)
        self.input_map.setDim([occ_grid_msg.info.width, occ_grid_msg.info.height])
        self.input_map.setOrigin([occ_grid_msg.info.origin.position.x, occ_grid_msg.info.origin.position.y])
        self.input_map.setData(occ_data)        
        rospy.loginfo("Occupancy map received")


    def computeDistanceMaps(self):
        """Compute the distance field associated to the occupancy map and its gradient"""
        # Compute distance and gradient maps
        #dist_map, grad_maps = computeDistMaps(filtered_map, obst_thresh, obst_d_offset=dist_field_offset)
        self.lock.acquire()
        try:
            rospy.loginfo('Acquired a lock')
            occ_map = self.occ_map_service().map
            self.occMapCallback(occ_map)
            dist_map, grad_maps = computeDistsScipy(self.input_map, 0.5, obst_d_offset=self.distance_offset)
            self.dist_map = dist_map
            if self.dist_occ_map is None :
                self.dist_occ_map = OccupancyGrid()
            self.dist_occ_map.header = occ_map.header
            self.dist_occ_map.info = occ_map.info
            self.dist_occ_map.info.resolution = self.dist_map.resolution
            self.dist_occ_map.info.origin.position.z = occ_map.info.origin.position.z - 1.
            self.dist_occ_map.info.width = self.dist_map.dim[0]
            self.dist_occ_map.info.height = self.dist_map.dim[1]
            self.dist_occ_map.data = (100*(1-(self.dist_map.data > 0)*(self.dist_map.data/np.max(self.dist_map.data)))).T.astype(np.uint8).flatten()
            self.dist_grad_x = grad_maps[0]
            self.dist_grad_y = grad_maps[1]
            self.publishMaps()
            rospy.loginfo("Distance computations finished")
        finally:
            rospy.loginfo('Released a lock')
            self.lock.release()
            
        

    def computeSkeletonMap(self):
        """Computes the thin skeleton of the distance field"""
        grad_x, grad_y = self.dist_grad_x.data, self.dist_grad_y.data
        dist_grad = np.array([grad_x, grad_y])
        gradient_flux_map = gradFlux(dist_grad)
        rospy.loginfo("Skeleton computations finished")
        self.skeleton_map = self.input_map.copy()
        self.skeleton_map.setData(fluxToSkeletonMap(gradient_flux_map, self.skeleton_flux_threshold))
        self.skeleton_map.dim = self.skeleton_map.data.shape


    def publishMaps(self):
        if self.dist_map is not None:
            dist_map_msg = envMapToROSMsg(self.dist_map, frame_id="map")
            grad_x_msg = envMapToROSMsg(self.dist_grad_x, frame_id="map")
            grad_y_msg = envMapToROSMsg(self.dist_grad_y, frame_id="map")
            #skel_map_msg = envMapToROSMsg(self.skeleton_map, frame_id="map")
            
            self.dist_map_publisher.publish(dist_map_msg)
            self.dist_as_occ_publisher.publish(self.dist_occ_map)
            self.dist_gradX_publisher.publish(grad_x_msg)
            self.dist_gradY_publisher.publish(grad_y_msg)
            #self.skel_map_publisher.publish(skel_map_msg)

    def distanceServiceCallback(self, request):
        if self.dist_map is None:
            self.computeDistanceMaps()
        response = GetDistanceResponse()
        response.distance = envMapToROSMsg(self.dist_map, frame_id="map")
        response.grad_x = envMapToROSMsg(self.dist_grad_x, frame_id="map")
        response.grad_y = envMapToROSMsg(self.dist_grad_y, frame_id="map")
        return response

    def distanceSkeletonServiceCallback(self, request):
        #self.computeDistanceMaps()
        self.computeSkeletonMap()
        response = GetDistanceSkeletonResponse()
        response.distance = envMapToROSMsg(self.dist_map, frame_id="map")
        response.skeleton = envMapToROSMsg(self.skeleton_map, frame_id="map")
        return response

    def update(self):
        self.computeDistanceMaps()


if __name__ == "__main__":   
    rospy.init_node("dist_server")

    time.sleep(0.5)

    # Parameters
    occupancy_map_service = rospy.get_param("~occupancy_map_service", "rtabmap/get_map")
    distance_offset = rospy.get_param("~distance_offset", 0.)
    skeleton_flux_threshold = rospy.get_param("~skeleton_flux_threshold", -1e-3)
    update_rate = rospy.get_param("~update_rate", 5)

    r = rospy.Rate(update_rate)
    rospy.wait_for_service(occupancy_map_service)
    dist_map_server = DistanceMapServer(occupancy_map_service, distance_offset, skeleton_flux_threshold)
    
    
    while not rospy.is_shutdown():
        r.sleep()
        dist_map_server.update()
    rospy.spin()