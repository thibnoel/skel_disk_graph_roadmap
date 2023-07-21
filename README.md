# skel_disk_graph_roadmap
Python implementation of the Skeleton-Disk-Graph Roadmap (SDGRM) planner.

*__Note__: This repo is intended for usage as a [ROS](https://wiki.ros.org/) metapackage containing 2 ROS packages responsible respectively for the planner itself and for utility code to support it, but the underlying code can easily be extracted and reused.*

The SDGRM planner is a deterministic roadmap, focused on safety and sparsity. Its principle is to rely on the occupancy data as input, computing the associated Signed Distance Field (SDF), from which a skeleton of the environment can be extracted.\
The skeleton is then sampled to place free-space bubbles which serve as the roadmap nodes; using such bubbles makes the distribution of the final roadmap nodes non-uniform and naturally adapted to local conditions (the nodes density is higher in areas of low SDF, i.e. close to the obstacles).

![Multi-path planning demo](figures/multipath_manim_figure.gif)
![Overview of the method](figures/sdg_construction_diagram.png)

## Requirements
For ease of development, our implementation relies on a few (Python) libraries as dependencies. 

**Both packages**:
- numpy 
- matplotlib
- rospy

**extended_navigation_mapping**:
- cv2: image processing (*dilation/erosion process*)
- scikit-image: more image processing (*skeleton thinning*)
- scipy: distance computations

**skeleton_disk_graph_roadmap**:
- networkx: graph data structures and generic methods
- pqdict: priority queues (*for radius-descending disk-graph construction*)

## Getting started
The code is provided as two separate ROS packages:
- [extended_navigation_mapping](./extended_navigation_mapping) first provides tools to extend the ROS data structures for mapping and implements the preprocessing steps necessary to our method (distance and skeleton maps extraction). It also contains all the navigation-related aspects for exploration, i.e. utility code for paths handling and our implementation of a non-linear path following controller (see the paper for reference).

- [skeleton_disk_graph_roadmap](./skeleton_disk_graph_roadmap) provides the core implementation of the roadmap construction method and of the exploration strategy as a ROS-agnostic python module, wrapped in a ROS package for usability in a more experimental context.

Example configuration and launch files are also provided in both packages.
For each package:
- the `src/nodes` folder contains executable ROS nodes
- the `config` folder contains corresponding example configurations 
- the `launch` folder contains example ROS launch files

Custom ROS messages and services are also defined in each package's `msg` and `srv` folder.

**Note**: if you only wish to use the method in a ROS-agnostic context, the essential necessary code is everything located in [`skeleton_disk_graph_roadmap/src/sdg_roadmap`](./skeleton_disk_graph_roadmap/src/sdg_roadmap) and the appropriate dependencies from [`extended_navigation_mapping/src/extended_mapping`](./extended_navigation_mapping/src/extended_mapping) (for skeleton extraction) and [`extended_navigation_mapping/src/navigation_utils`](./extended_navigation_mapping/src/navigation_utils) (for paths utilities).

### Running the example script
A Python-only [demo script](./skeleton_disk_graph_roadmap/src/examples/sdg_roadmap_demo.py) is available which you can simply run using 
````
python3 sdg_roadmap_demo.py
````
It demonstrates all the steps of the method (skeleton extraction, disk-graph construction, path planning, exploration path selection) on a standard [occupancy map](./skeleton_disk_graph_roadmap/environments/env_intellab.png).

### Running the planner in ROS
(WIP section)

## Cite this work
The code in this repo corresponds to the method presented in the following paper (*currently under review*):
> *Skeleton Disk-Graph Roadmap: a Sparse Deterministic Roadmap for Safe 2D Navigation and Exploration* - T.Noël, A.Lehuger, E.Marchand, F.Chaumette
