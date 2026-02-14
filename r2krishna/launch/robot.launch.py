import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    urdf = os.path.expanduser('~/ros2_ws/src/r2krishna/urdf/R2krishna.urdf')
    with open(urdf, 'r') as infp: robot_desc = infp.read()

    return LaunchDescription([
        Node(
            package='ros_gz_sim', executable='create',
            arguments=['-topic', 'robot_description', '-name', 'r2krishna', '-x', '5', '-y', '2', '-z', '0.5'],
            output='screen'
        ),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen', parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=['/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist', '/model/r2krishna/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry', '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V'], output='screen')
    ])
