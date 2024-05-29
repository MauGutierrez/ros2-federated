#!/bin/bash

# Source ROS
echo 'Source ROS 2 package'
source /opt/ros/humble/setup.bash

# Compile ros2-federated project
echo 'Compilation of ROS 2 project'
colcon build --packages-ignore ros2_federated_agents

# Create results directory to store models and other stuff
echo 'Delete already used Environment Variable'
sed -i '/EXPERIMENT_NAME/d' ~/.bashrc
echo 'export EXPERIMENT_NAME="Unity RL testing"' >> ~/.bashrc
source ~/.bashrc

mkdir -p results/$EXPERIMENT_NAME
mkdir -p results/$EXPERIMENT_NAME/history
mkdir -p results/$EXPERIMENT_NAME/images
mkdir -p results/$EXPERIMENT_NAME/time