import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('ros2_federated'),
        'config/params.yaml'
    )

    server = Node(
        package="ros2_federated",
        name="federated_server",
        executable="main",
        output='screen',
        parameters=[config]
    )
    
    ld.add_action(server)

    return ld