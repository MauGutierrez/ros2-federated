#!/bin/bash
echo '/opt/ros/jazzy/setup.bash'
source /opt/ros/jazzy/setup.bash

echo './install/setup.bash'
source ./install/setup.bash

echo 'Init ROS2 Server'
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=$1;