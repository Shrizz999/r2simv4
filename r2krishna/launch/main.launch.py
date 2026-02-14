import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_r2krishna = get_package_share_directory('r2krishna')

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=os.path.expanduser('~/ros2_ws/src')),
        
        # 1. Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': '-r empty.sdf'}.items(),
        ),
        
        # 2. RViz
        Node(
            package='rviz2', executable='rviz2',
            arguments=['-d', os.path.join(pkg_r2krishna, 'config', 'view_bot.rviz')],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        # 3. Modules
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.expanduser('~/ros2_ws/src/r2krishna/launch/arena.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.expanduser('~/ros2_ws/src/r2krishna/launch/robot.launch.py')))
    ])
