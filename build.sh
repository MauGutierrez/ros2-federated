#!/bin/bash

# Source ROS
echo 'Source ROS 2 package'
source /opt/ros/humble/setup.bash

# Compile ros2-federated project
echo 'Compilation of ROS 2 project'
colcon build