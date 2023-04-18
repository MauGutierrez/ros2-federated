echo '/opt/ros/humble/setup.bash'
source /opt/ros/humble/setup.bash

echo 'colcon build'
colcon build

echo '. install/setup.bash'
source ./install/setup.bash