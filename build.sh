#!/bin/bash

# Source ROS
echo '/opt/ros/humble/setup.bash'
source /opt/ros/humble/setup.bash

# Compile ros2-federated project
echo 'colcon build'
colcon build

# Create results directory to store models and other stuff
mkdir -p results
mkdir -p results/history
mkdir -p results/images