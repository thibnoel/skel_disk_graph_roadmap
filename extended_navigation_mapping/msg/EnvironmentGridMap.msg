# Generalization of the nav_msgs/OccupancyGrid ROS message
# The data field is float rather than int and is intended to represent less specific data (distance, information content...)
std_msgs/Header header
nav_msgs/MapMetaData info
float32[] data