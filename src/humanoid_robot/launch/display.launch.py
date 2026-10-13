import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_robot')
    g1_share = get_package_share_directory('g1_description')
    default_urdf = os.path.join(g1_share, 'urdf', 'g1_23dof.urdf')
    default_rviz = os.path.join(pkg_share, 'rviz', 'humanoid.rviz')

    model_arg = DeclareLaunchArgument(
        'model',
        default_value=default_urdf,
        description='Path to the humanoid xacro/urdf file',
    )

    robot_description = Command(['xacro ', LaunchConfiguration('model')])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', default_rviz],
    )

    lidar_node = Node(
        package='humanoid_robot',
        executable='lidar_simulator',
        name='lidar32_simulator',
        output='screen',
    )

    nav_node = Node(
        package='humanoid_robot',
        executable='navigate_to_goal',
        name='navigate_to_goal',
        output='screen',
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher_node,
        rviz_node,
        lidar_node,
        nav_node,
    ])
