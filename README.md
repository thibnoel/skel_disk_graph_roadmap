# skel_disk_graph_roadmap
Python implementation of the Skeleton-Disk-Graph Roadmap (SDGRM) planner.

The SDGRM planner is a deterministic roadmap, focused on safety and sparsity. Its principle is to rely on the occupancy data as input, computing the associated Signed Distance Field (SDF), from which a skeleton of the environment can be extracted.
The skeleton is then sampled to place free-space bubbles which serve as the roadmap nodes; using such bubbles makes the distribution of the final roadmap nodes non-uniform and naturally adapted to local conditions (the nodes density is higher in areas of low SDF, i.e. close to the obstacles).

# Requirements
- numpy
- networkx
- cv2
- pqdict
- scipy

# Getting started
(TODO) Provide clear example scripts\
(TODO) detail usage in a real use-case

# Cite this work
The code in this repo corresponds to the method presented in the following paper:
> *Skeleton Disk-Graph Roadmap: a Sparse Deterministic Roadmap for Safe 2D Navigation and Exploration* - T.Noël, A.Lehuger, E.Marchand, F.Chaumette
