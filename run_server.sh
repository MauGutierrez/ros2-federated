echo 'Cleanning the ROS project'
. clean.sh

echo 'Compiling the ROS project'
. build.sh

echo 'Sourcing the installation'
. install.sh

echo 'Starting Federated Server'

server_mode=$1
params_path=$2

ros2 run ros2_federated_server main --mode=$server_mode --ros-args --params-file $params_path 