import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('r2krishna')
    default_rviz_config_path = os.path.join(pkg_share, 'config', 'view_robot.rviz')

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', default_rviz_config_path],
            parameters=[{'use_sim_time': True}]
        )
    ])