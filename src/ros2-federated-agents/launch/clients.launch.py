import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def generate_launch_description():
    res = []

    config = os.path.join(
        get_package_share_directory('ros2_federated_agents'),
        'config/params.yaml'
    )
    
    client_1 = Node(
        package="ros2_federated_agents",
        name="client_1",
        executable="client_1",
        output='screen',
        parameters=[config]
    )
    res.append(client_1)

    client_2 = Node(
        package="ros2_federated_agents",
        name="client_2",
        executable="client_2",
        output='screen',
        parameters=[config]
    )
    res.append(client_2)

    client_3 = Node(
        package="ros2_federated_agents",
        name="client_3",
        executable="client_3",
        output='screen',
        parameters=[config]
    )
    res.append(client_3)

    client_4 = Node(
        package="ros2_federated_agents",
        name="client_4",
        executable="client_4",
        output='screen',
        parameters=[config]
    )
    res.append(client_4)

    return LaunchDescription(res)