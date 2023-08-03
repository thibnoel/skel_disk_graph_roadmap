import rospy
import actionlib
from sdg_roadmap.skel_disk_graph_provider import *
from sdg_roadmap.ros_conversions import *
from extended_mapping.map_processing import EnvironmentMap
from extended_mapping.ros_conversions import *
from navigation_utils.ros_conversions import *

from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_msgs.msg import Float64, Int32
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetMap
from extended_navigation_mapping.msg import FollowPathAction, FollowPathGoal, FollowPathActionResult
from extended_navigation_mapping.srv import GetDistance, GetDistanceSkeleton
from skeleton_disk_graph_roadmap.msg import DiskGraph, DiskGraphNode, DiskGraphEdge
from skeleton_disk_graph_roadmap.srv import PlanPath, PlanPathResponse, GetDiskGraph, GetDiskGraphResponse, NavigateTo, NavigateToResponse


class SDGPlanningComponent:
    """
    ROS wrapper to provide a high-level planning service using the skel. disk-graph roadmap
    """
    def __init__(self, sdg_server, visualizer = None):
        self.sdg_server = sdg_server
        self.visualizer = visualizer
        # Publishers 
        self.path_publisher = rospy.Publisher("~planned_path", Path, queue_size=1, latch=True)
        # Services
        self.path_planning_service = rospy.Service("~plan_path", PlanPath, self.pathPlanningCallback)
        
    def pathPlanningCallback(self, path_planning_request):
        """
        Computes and returns a feasible path as a PlanPathResponse on a new path planning request
        """
        self.sdg_server.updatePlanner(None)
        start = self.sdg_server.agent_pos_listener.get2DAgentPos()
        goal = np.array([path_planning_request.goal.x,
                        path_planning_request.goal.y])
        world_path = self.sdg_server.sdg_provider.getWorldPath(start, goal, postprocessed=True)
        if world_path is None:
            rospy.logwarn("Planning failed (source : {}, target: {})".format(start, goal))
            return PlanPathResponse()
        rospy.loginfo("Planning succesful (source : {}, target: {})".format(start, goal))
        return self.constructPlanPathMsg(world_path)

    def constructPlanPathMsg(self, waypoints_path, send_viz_msg=True):
        """
        Constructs a ROS PlanPathResponse from a WaypointsPath returned by the planner
        """
        path_msg = waypointsPathToRosPath(waypoints_path)
        self.path_publisher.publish(path_msg)
        if self.visualizer is not None and send_viz_msg:
            self.visualizer.publishPlanPathVizMsg(waypoints_path, height_offset=0.02, nodes_subsamp_ratio=0.2)
        response_path = PlanPathResponse()
        response_path.path = path_msg
        return response_path


class SDGNavigationComponent:
    """
    ROS wrapper to provide a high-level navigation service using the skel. disk-graph roadmap
    """
    def __init__(self, sdg_server, sdg_planning_comp, path_follower_node_name, path_safety_distance):
        self.sdg_server = sdg_server
        self.sdg_planning_comp = sdg_planning_comp
        self.path_safety_distance = path_safety_distance
        # Subscribers
        self.pf_result_subscriber = rospy.Subscriber(
            path_follower_node_name + "/follow_path/result", FollowPathActionResult, self.pathFollowingResultCallback)
        # Publishers
        self.cancel_publisher = rospy.Publisher(
            path_follower_node_name + "/follow_path/cancel", GoalID, queue_size=1)
        # Action server client
        self.follow_path_client = actionlib.SimpleActionClient(
            path_follower_node_name + "/follow_path", FollowPathAction)
        # Service provider 
        self.nav_service = rospy.Service("~navigate_to", NavigateTo, self.navToCallback)
        self.interrupt_service= rospy.Service("~interrupt_nav", Empty, self.interruptCallback)

        # State
        self.is_following_path = False
        self.current_target = None
        self.current_path = None

    def pathFollowingResultCallback(self, pf_result_msg):
        """
        Callback triggered by the FollowPathActionResult subscriber - informs on path following result and updates state
        """
        rospy.loginfo(str(pf_result_msg))
        self.is_following_path = False

    def callPathFollowingDetached(self, goal_path):
        """
        On PlanPath request, calls the corresponding path following service to execute the requested plan
        """
        self.current_path = rosPathToWaypointsPath(goal_path.path)
        rospy.loginfo("Start path following")
        self.follow_path_client.send_goal(goal_path)
        self.is_following_path = True

    def checkPathValidity(self):
        """
        Check the validity of the current path in terms of safety distance 
        """
        if self.current_path is not None :
            dist_map = envMapFromRosEnvGridMapMsg(self.sdg_server.distance_serv_proxy().distance)
            for k, p in enumerate(self.current_path.waypoints):
                d = dist_map.valueAt(p)
                checkColl = d > self.path_safety_distance
                if not checkColl:
                    rospy.logwarn("Interrupting path following - Path safety distance threshold violated")
                    cancel_msg = GoalID()
                    self.cancel_publisher.publish(cancel_msg)

    def navToCallback(self, nav_req):
        """
        On NavigateTo request, requests a plan from the planner and executes it by calling the path following service
        """
        path_request = PlanPath()
        path_request.goal = nav_req.goal
        rospy.loginfo("Received nav. request to {}".format(np.array([nav_req.goal.x, nav_req.goal.y])))
        path = self.sdg_planning_comp.pathPlanningCallback(path_request)
        self.callPathFollowingDetached(path)
        return NavigateToResponse()

    def interruptCallback(self, msg):
        """
        On Empty request, interrupts the current navigation plan
        """
        cancel_msg = GoalID()
        self.cancel_publisher.publish(cancel_msg)
        rospy.loginfo("Interrupting path following - Requested")
        return EmptyResponse()
    
    def update(self):
        """
        Method called repeatedly in the main node
        """
        self.checkPathValidity()        


class SDGExplorationComponent:
    """
    ROS wrapper to execute autonomous exploration using the skel. disk-graph roadmap
    """
    def __init__(self, sdg_server, sdg_planning_comp, sdg_nav_comp, sdg_strategy_parameters, occupancy_map_service, start_paused=True, visualizer=None):
        self.sdg_server = sdg_server
        self.sdg_planning_comp = sdg_planning_comp
        self.sdg_nav_comp = sdg_nav_comp
        self.visualizer = visualizer

        self.sdg_exploration_path_provider = SDGExplorationPathProvider(
            self.sdg_server.sdg_provider,
            sdg_strategy_parameters["unkn_occ_range"],
            sdg_strategy_parameters["frontiers_max_known_ratio"],
            sdg_strategy_parameters["search_dist_increment"],
            sdg_strategy_parameters["path_cost_parameters"]
        )
        # Service proxy
        self.occupancy_map_service = rospy.ServiceProxy(occupancy_map_service, GetMap)
        # Service provider
        self.pause_service = rospy.Service("~exploration/pause_resume", Empty, self.pauseCallback)
        self.preview_service = rospy.Service("~exploration/preview_path", Empty, self.previewCallback)
        # State
        self.paused = start_paused
        self.current_path = None
        self.current_target_pos = None

    def pauseCallback(self, msg):
        if not self.paused:
            cancel_msg = GoalID()
            rospy.loginfo("Pausing exploration")
            self.sdg_nav_comp.cancel_publisher.publish(cancel_msg)
        if self.paused:
            rospy.loginfo("Resuming exploration")
        self.paused = not(self.paused)
        return EmptyResponse()

    def previewCallback(self, msg):
        """
        Purely for viz. purposes for now
        Show a preview of the path selection (useful while paused)
        """
        plan = self.selectBestPath(best_path_viz_msg=False)
        if plan is None:
            rospy.logwarn("Path preview - No valid exploration path found")
        return EmptyResponse()

    def selectBestPath(self, best_path_viz_msg=True):
        """
        Selects the current best exploration path and returns it as a navigation request
        """
        start = self.sdg_server.agent_pos_listener.get2DAgentPos()
        dist_map = envMapFromRosEnvGridMapMsg(self.sdg_server.distance_serv_proxy().distance)
        #self.replan_pos.append(start)
        self.sdg_server.updatePlanner(None)
        frontiers_paths = self.sdg_exploration_path_provider.getFrontiersPaths(start, envMapFromRosEnvGridMapMsg(self.occupancy_map_service().map))
        if not len(frontiers_paths):
            rospy.logwarn("No frontiers found")
            return None
        best_id, best_path = self.sdg_exploration_path_provider.selectExplorationPath(frontiers_paths)
        if best_path is None:
            return None
        self.current_target_pos = best_path['path'].waypoints[-1]
        best_path = best_path['path']
        self.current_path = best_path

        if self.visualizer is not None:
            self.visualizer.publishPathSelectionViz(frontiers_paths, height_offset=0.05, outline=True)
        return self.sdg_planning_comp.constructPlanPathMsg(best_path, send_viz_msg=best_path_viz_msg)

    def interruptOnInvalidTarget(self):
        """
        Checks the current target validity and interrupt the current plan if it is invalid.
        Two criteria are considered:
            - target bubble radius (must be > minimal radius)
            - target bubble unknown covered surface (must be > 0)
        """
        occ_map = envMapFromRosEnvGridMapMsg(self.occupancy_map_service().map)
        dist_map = envMapFromRosEnvGridMapMsg(self.sdg_server.distance_serv_proxy().distance)
        if self.current_target_pos is not None :
            target_rad = dist_map.valueAt(self.current_target_pos)
            if self.visualizer is not None:
                self.visualizer.publishCurrTargetViz(self.current_target_pos, target_rad, height_offset=0.05)
            if target_rad < self.sdg_server.sdg_provider.parameters_dict["min_pbubbles_rad"] :
                cancel_msg = GoalID()
                rospy.loginfo("Trigger exploration replan - target bubble became too small")
                self.sdg_nav_comp.cancel_publisher.publish(cancel_msg)
                return
            
            # Interrupt if target becomes known
            if not self.sdg_exploration_path_provider.checkFrontierValidity(self.current_target_pos, occ_map, dist_map):
                self.current_path = None
                rospy.loginfo("Trigger exploration replan - target bubble fully known")
                cancel_msg = GoalID()
                self.sdg_nav_comp.cancel_publisher.publish(cancel_msg)
                return

    def update(self):
        """
        Method called repeatedly in the main node
        """
        # Set current path to None to trigger replan if navigation was interrupted
        if self.paused:
            return
        if not self.sdg_nav_comp.is_following_path:
            self.current_path = None 
        # Select new path or check target validity during nav.
        if self.current_path is None:
            plan_path = self.selectBestPath(best_path_viz_msg=True)
            if plan_path is None:
                rospy.logwarn("No valid exploration path found")
                return
            self.sdg_nav_comp.callPathFollowingDetached(plan_path)
        else:
            self.interruptOnInvalidTarget()