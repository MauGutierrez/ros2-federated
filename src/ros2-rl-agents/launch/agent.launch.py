import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('ros2_rl_agents'),
        'config/agents.yaml'
    )

    # agent_node = Node(
    #     package="ros2_rl_agents",
    #     name="agent_1",
    #     executable="run_agent",
    #     output='screen',
    #     parameters=[config]
    # )
    
    # ld.add_action(agent_node)

    # return ld

    return LaunchDescription([
        Node(
            package="ros2_rl_agents",
            name="agent_1",
            executable="test_agent",
            output='screen',
            parameters=[config]
        ),
        Node(
            package="ros2_rl_agents",
            name="agent_2",
            executable="test_agent",
            output='screen',
            parameters=[config]
        ),
        Node(
            package="ros2_rl_agents",
            name="agent_3",
            executable="test_agent",
            output='screen',
            parameters=[config]
        ),
        Node(
            package="ros2_rl_agents",
            name="agent_4",
            executable="test_agent",
            output='screen',
            parameters=[config]
        ),
        # Node(
        #     package="ros2_rl_agents",
        #     name="agent_5",
        #     executable="test_agent",
        #     output='screen',
        #     parameters=[config]
        # ),
        # Node(
        #     package="ros2_rl_agents",
        #     name="agent_6",
        #     executable="test_agent",
        #     output='screen',
        #     parameters=[config]
        # ),
        # Node(
        #     package="ros2_rl_agents",
        #     name="agent_7",
        #     executable="test_agent",
        #     output='screen',
        #     parameters=[config]
        # ),
        # Node(
        #     package="ros2_rl_agents",
        #     name="agent_8",
        #     executable="test_agent",
        #     output='screen',
        #     parameters=[config]
        # )    
    ])