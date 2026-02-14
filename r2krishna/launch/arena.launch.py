import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    urdf = os.path.expanduser('~/ros2_ws/src/arena_viz/models/DDRena/DDRena.urdf')
    return LaunchDescription([
        Node(
            package='ros_gz_sim', executable='create',
            arguments=['-file', urdf, '-name', 'arena', '-x', '0', '-y', '0', '-z', '0'],
            output='screen'
        )
    ])
