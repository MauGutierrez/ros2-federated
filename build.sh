#!/bin/bash

# Source ROS
echo '/opt/ros/humble/setup.bash'
source /opt/ros/humble/setup.bash

# Compile ros2-federated project
echo 'colcon build'
colcon build

# Create results directory to store models and other stuff
echo 'Delete already used Environment Variable'
sed -i '/EXPERIMENT_NAME/d' ~/.bashrc
echo 'export EXPERIMENT_NAME="4_agent_1_iterations_20_epoch"' >> ~/.bashrc
source ~/.bashrc

mkdir -p results/$EXPERIMENT_NAME
mkdir -p results/$EXPERIMENT_NAME/history
mkdir -p results/$EXPERIMENT_NAME/images
mkdir -p results/$EXPERIMENT_NAME/time