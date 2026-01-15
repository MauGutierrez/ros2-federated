echo 'Starting Federated Server'

server_mode=$1
params_path=$2

ros2 run ros2_federated_server main --mode=$server_mode --ros-args --params-file $params_path 