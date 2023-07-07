from sdg_roadmap.sdg_roadmap_utils import *


def getFrontierNodes(bubble_nodes, occ_map, min_unkn_ratio, unkn_range=[-1,-1], unkn_max_ratio=1):
    frontier_nodes = []
    for k,b in enumerate(bubble_nodes):
        '''
        bmask = circularMask(occ_map.dim, occ_map.worldToMapCoord(b.pos),
                                b.bubble_rad/occ_map.resolution)
        unkn_mask = (occ_map.data >= unkn_range[0])
        unkn_mask = unkn_mask*(occ_map.data <= unkn_range[1])
        unknown_in_bubble = unkn_mask*bmask
        b_unkn_ratio = np.sum(unknown_in_bubble)/np.sum(bmask)
        '''
        b_cov, has_obst = b.computeBubbleCoverage(occ_map, unkn_range=unkn_range)
        if 1-b_cov > min_unkn_ratio and 1-b_cov < unkn_max_ratio and not has_obst:
            frontier_nodes.append(b)
    return frontier_nodes


def bubbleUnknownCoverageReward(bubble_node, occ_map, unkn_range=[-1,-1]):
    bmask = circularMask(occ_map.dim, occ_map.worldToMapCoord(bubble_node.pos),
                                bubble_node.bubble_rad/occ_map.resolution)
    unkn_mask = (occ_map.data >= unkn_range[0])
    unkn_mask = unkn_mask*(occ_map.data <= unkn_range[1])
    unknown_in_bubble = unkn_mask*bmask
    return np.pi*(bubble_node.bubble_rad*bubble_node.bubble_rad)*np.sum(unknown_in_bubble)/np.sum(bmask)


def pathEnergyCost(waypoints_path):
    return WaypointsPath(waypoints_path.waypoints[:-1]).getTotalLength()

def pathNarrowPassagesCost(nodes_radii, d_threshold):
    return np.sum(nodes_radii[:-1] < d_threshold)

def pathSharedNodesReward(all_paths_ids):
    all_ids = np.concatenate([pid for pid in all_paths_ids])
    unique_ids = np.unique(all_ids)
    nodes_sharing_count = {}
    for ind in unique_ids:
        nodes_sharing_count[ind] = np.sum(all_ids == ind)
        if nodes_sharing_count[ind] == len(all_paths_ids):
            nodes_sharing_count[ind] = 0
        nodes_sharing_count[ind] *= nodes_sharing_count[ind]
    all_rewards = []
    for p in all_paths_ids:
        reward = 0
        for i in p:
            reward += nodes_sharing_count[i]
        all_rewards.append(reward/len(p))
        #for kp, other_ids in enumerate(all_paths_ids):
        #    if i in other_ids:
        #        reward += 1
    return all_rewards

def pathUnknownCoverageReward(path_nodes, occ_map, unkn_range=[-1,-1]):
    reward = 0
    bpos, brad = [], []
    for pn in path_nodes:
        #reward += bubbleUnknownCoverageReward(pn, occ_map, unkn_range=unkn_range)
        bpos.append(pn.pos)
        brad.append(pn.bubble_rad)
    bmask = bubblesMask(occ_map.dim, bpos, brad, radInflationMult=1.0, radInflationAdd=0.0)
    unkn_mask = (occ_map.data >= unkn_range[0])
    unkn_mask = unkn_mask*(occ_map.data <= unkn_range[1])
    unknown_in_bubble = unkn_mask*bmask
    reward = np.sum(unknown_in_bubble)
    return reward

class SDGFrontiersStrategy:
    def __init__(self, 
            sdg_roadmap,
            min_frontier_unkn_ratio,
            max_frontier_unkn_ratio,
            unkn_occ_range,
            narrowness_cost_d_threshold, 
            max_dist_travel_compromise,
            unknown_max_coverage,
            frontiers_min_coverage
        ):
        self.sdg_roadmap = sdg_roadmap
        self.min_frontier_unkn_ratio = min_frontier_unkn_ratio
        self.max_frontier_unkn_ratio = max_frontier_unkn_ratio
        self.unkn_occ_range = unkn_occ_range
        self.narrowness_cost_d_threshold = narrowness_cost_d_threshold
        self.max_dist_travel_compromise = max_dist_travel_compromise
        self.unknown_max_coverage = unknown_max_coverage
        self.frontiers_min_coverage = frontiers_min_coverage
        
        self.max_search_dist = 10
        self.max_search_dist_increase_rate = 1.55
        self.max_search_dist_decrease_rate = 0.99
        self.max_search_iter = 20

        self.prev_frontiers_pos = []


    def updateRoadmap(self, roadmap):
        self.sdg_roadmap = roadmap
    
    def getValidFrontiers(self, pos, occ_map): #, unkn_ratio, unkn_range=[-1,-1], unkn_max_ratio=1):
        nodes = [self.sdg_roadmap.graph.nodes[i]['node_obj'] for i in self.sdg_roadmap.graph.nodes]
        #frontiers = getFrontierNodes(nodes, occ_map, self.min_frontier_unkn_ratio, unkn_range=self.unkn_occ_range, unkn_max_ratio=self.max_frontier_unkn_ratio)
        #frontiers_ids = [f.id for f in frontiers]
        
        frontiers_ids = self.sdg_roadmap.segmentFrontiers(pos, occ_map, self.unkn_occ_range, unkn_threshold=self.unknown_max_coverage, frontiers_threshold=self.frontiers_min_coverage, max_search_dist=self.max_search_dist)
        #frontiers_ids = self.sdg_roadmap.last_added_ids
        frontiers = [self.sdg_roadmap.graph.nodes[fid]['node_obj'] for fid in frontiers_ids]
        if not len(frontiers):
            self.max_search_dist *= self.max_search_dist_increase_rate
            return None, None
        frontiers_paths = self.sdg_roadmap.getNodesPaths(pos, ids=frontiers_ids)
        if frontiers_paths is None :
            return None, None
        frontiers_paths = {fp:frontiers_paths[fp] for fp in frontiers_paths}
        valid_frontiers_ids = [frontiers_paths[fp]['nodes_ids'][-1] for fp in frontiers_paths]
        return valid_frontiers_ids, frontiers_paths

    def newGetValidFrontiers(self, pos, occ_map, dist_map, search_dist_increment):
        '''
        frontiers = []
        self.max_search_dist = 10
        abs_max_search_dist = 10*occ_map.resolution*np.sqrt(occ_map.dim[0]*occ_map.dim[0] + occ_map.dim[1]*occ_map.dim[1])
        while (not len(frontiers)) and self.max_search_dist < abs_max_search_dist:
            frontiers_ids = self.sdg_roadmap.segmentFrontiers(pos, occ_map, self.unkn_occ_range, unkn_threshold=self.unknown_max_coverage, frontiers_threshold=self.frontiers_min_coverage, max_search_dist=self.max_search_dist)
            frontiers = [self.sdg_roadmap.graph.nodes[fid]['node_obj'] for fid in frontiers_ids]
            if len(frontiers):
                frontiers_paths = self.sdg_roadmap.getNodesPaths(pos, ids=frontiers_ids)
                if frontiers_paths is None :
                    return None, None
                frontiers_paths = {fp:frontiers_paths[fp] for fp in frontiers_paths}
                valid_frontiers_ids = [frontiers_paths[fp]['nodes_ids'][-1] for fp in frontiers_paths]
                return valid_frontiers_ids, frontiers_paths
            self.max_search_dist *= self.max_search_dist_increase_rate
        #self.max_search_dist = 10
        '''
        #self.refilterPrevFrontiers(pos, occ_map, dist_map)
        frontiers_ids, _ = self.sdg_roadmap.searchClosestFrontiers(pos, occ_map, self.unkn_occ_range, self.unknown_max_coverage, search_dist_increment)
        #frontiers_ids.extend([self.sdg_roadmap.getClosestNodes(p, 1)[0] for p in self.prev_frontiers_pos])
        frontiers = [self.sdg_roadmap.graph.nodes[fid]['node_obj'] for fid in frontiers_ids]
        if len(frontiers):
            frontiers_paths = self.sdg_roadmap.getNodesPaths(pos, ids=frontiers_ids)
            if frontiers_paths is None :
                return None, None
            frontiers_paths = {fp:frontiers_paths[fp] for fp in frontiers_paths}
            valid_frontiers_ids = [frontiers_paths[fp]['nodes_ids'][-1] for fp in frontiers_paths]
            return valid_frontiers_ids, frontiers_paths
        return None, None

    def filterFrontiersByTravelDist(self, frontiers_ids, frontiers_paths):
        frontiers_paths_lengths = [frontiers_paths[p]['nodes_path_length'] for p in frontiers_paths]
        max_travel_dist = self.max_dist_travel_compromise + np.min(frontiers_paths_lengths)
        valid_frontiers_paths = {fp: frontiers_paths[fp] for fp in frontiers_paths if frontiers_paths[fp]['nodes_path_length'] <= max_travel_dist}
        valid_frontiers_ids = [frontiers_paths[fp]['nodes_ids'][-1] for fp in valid_frontiers_paths]
        return valid_frontiers_ids, valid_frontiers_paths

    def refilterPrevFrontiers(self, pos, occ_map, dist_map):
        updated_prev_frontiers = []
        # INCATIVE FOR NOW
        start_id = self.sdg_roadmap.getClosestNodes(pos, 1)[0]
        for k,p in enumerate(self.prev_frontiers_pos):
            if self.checkFrontierValidity(p, occ_map, dist_map):
                p_id = self.sdg_roadmap.getClosestNodes(p, 1)[0]
                if nx.has_path(self.sdg_roadmap.graph, start_id, p_id):
                    if not np.linalg.norm(pos - p) < 1:
                        updated_prev_frontiers.append(p)
        
        self.prev_frontiers_pos = updated_prev_frontiers
                

    def checkFrontierValidity(self, node_pos, occ_map, dist_map):
        #node = self.sdg_roadmap.graph.nodes[node_id]['node_obj']
        new_node_rad = dist_map.valueAt(node_pos)
        bmask = circularMask(occ_map.dim, occ_map.worldToMapCoord(node_pos), 0.9*new_node_rad/occ_map.resolution - 1)
        unkn_mask = (occ_map.data >= self.unkn_occ_range[0])
        unkn_mask = unkn_mask*(occ_map.data <= self.unkn_occ_range[1])
        unknown_in_bubble = unkn_mask*bmask
        if np.sum(unknown_in_bubble) > 0: #/np.sum(bmask) > 0:#and np.sum(obst_bmask*obst_mask) == 0:
            return True
        return False
        
        '''
        coverage, has_obst = node.computeBubbleCoverage(occ_map, unkn_range=unkn_range, obst_thresh=obst_thresh)
        if coverage < 1-unkn_ratio :#and not has_obst:
            return True
        return False
        '''
    
    def getFrontiersPathsCosts(self, frontiers_ids, frontiers_paths, occ_map):
        """
        Need to use path as list of bubble indices and thus modify getPathNodes... find a good way
        """
        costs = {}
        
        waypoints_paths = [WaypointsPath(frontiers_paths[p]['postprocessed']) for p in frontiers_paths]
        paths_nodes_radii = [[self.sdg_roadmap.graph.nodes[ind]['node_obj'].bubble_rad for ind in frontiers_paths[p]['nodes_ids']] for p in frontiers_paths]
        all_paths_nodes_id = [frontiers_paths[p]['nodes_ids'] for p in frontiers_paths]

        energy_costs = [pathEnergyCost(wp_path) for wp_path in waypoints_paths]
        narrow_costs = [pathNarrowPassagesCost(np.array(pradii), self.narrowness_cost_d_threshold) for pradii in paths_nodes_radii]
        shared_nodes_rewards = pathSharedNodesReward(all_paths_nodes_id)
        coverage_rewards = [
            pathUnknownCoverageReward([
                self.sdg_roadmap.graph.nodes[ind]['node_obj'] for ind in pid
            ], occ_map, unkn_range=self.unkn_occ_range) for pid in all_paths_nodes_id
        ]

        energy_costs = np.array(energy_costs)
        #energy_costs = (energy_costs - np.min(energy_costs))/(np.max(energy_costs) - np.min(energy_costs))
        
        #narrow_costs = np.array(narrow_costs)
        #narrow_costs = (narrow_costs - np.min(narrow_costs))/(np.max(narrow_costs) - np.min(narrow_costs))
        
        shared_nodes_rewards = np.array(shared_nodes_rewards)
        shared_nodes_rewards = (shared_nodes_rewards - np.min(shared_nodes_rewards))/(np.max(shared_nodes_rewards) - np.min(shared_nodes_rewards))
        
        coverage_rewards = np.array(coverage_rewards)
        coverage_rewards = (coverage_rewards - np.min(coverage_rewards))/(np.max(coverage_rewards) - np.min(coverage_rewards))

        for k,f in enumerate(frontiers_ids):
            costs[f] = {}
            costs[f]['energy_cost'] = energy_costs[k]
            #costs[f]['narrow_cost'] = narrow_costs[k]
            costs[f]['sh_nodes_reward'] = shared_nodes_rewards[k]
            costs[f]['coverage_reward'] = coverage_rewards[k]

        return costs

    def selectPath(self, frontiers_paths, frontiers_paths_costs):
        def _pathSelectionCost(path_costs, e_cost_extrema):
            a_cov = 1
            a_sh = 0.
            a_energy = 2
            a_narrow = 0
            norm_e_cost = (path_costs['energy_cost'] - e_cost_extrema[0])/(e_cost_extrema[1] - e_cost_extrema[0])
            #return (a_cov*path_costs['coverage_reward'] + a_sh*path_costs['sh_nodes_reward'])/(1 + a_energy*norm_e_cost + a_narrow*path_costs['narrow_cost'])
            return a_cov*path_costs['coverage_reward'] + a_sh*path_costs['sh_nodes_reward'] - a_energy*norm_e_cost #- a_narrow*path_costs['narrow_cost']

        path_dists_costs = [frontiers_paths_costs[p]['energy_cost'] for p in frontiers_paths_costs]
        max_travel_dist = self.max_dist_travel_compromise + np.min(path_dists_costs)
        print(path_dists_costs)
        valid_frontiers_paths = [fp for fp in frontiers_paths if frontiers_paths_costs[fp]['energy_cost'] <= max_travel_dist]
        #if not len(valid_frontiers_paths):
        #    return None
        #valid_frontiers_rewards = {vfp: frontiers_paths_costs[vfp]['coverage_reward'] for vfp in valid_frontiers_paths}
        valid_frontiers_rewards = {vfp: _pathSelectionCost(frontiers_paths_costs[vfp], [np.min(path_dists_costs), np.max(path_dists_costs)]) for vfp in valid_frontiers_paths}
        return frontiers_paths[max(valid_frontiers_rewards, key=valid_frontiers_rewards.get)]

    def displayFrontiersPathsCosts(self, frontiers_paths, frontiers_paths_costs, occ_map, cost_key=None, markers_size = 60, paths_cmap = plt.cm.viridis):
        occ_map.display(cmap=plt.cm.binary)
        for k, p in enumerate(frontiers_paths_costs):
            cost = frontiers_paths_costs[p]
            path = np.array(frontiers_paths[p]['postprocessed'])

            if cost_key is None:
                path_color='blue'
                path_alpha = 0.5
            else:
                path_color = paths_cmap(cost[cost_key])
                path_alpha = 1

            plt.plot(path[:,0], path[:,1], marker='', lw=2, color=path_color,alpha=path_alpha)
            plt.scatter(path[-1,0], path[-1,1], c=path_color, s=0.5*markers_size)




