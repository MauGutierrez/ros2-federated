import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    res = []

    config = os.path.join(
        get_package_share_directory('ros2_federated'),
        'config/params.yaml'
    )
    
    client_1 = Node(
        package="ros2_federated",
        name="client_1",
        executable="client_1",
        output='screen',
        parameters=[config]
    )
    res.append(client_1)

    client_2 = Node(
        package="ros2_federated",
        name="client_2",
        executable="client_2",
        output='screen',
        parameters=[config]
    )
    res.append(client_2)

    return LaunchDescription(res)