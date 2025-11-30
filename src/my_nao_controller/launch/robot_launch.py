"""
robot_launch.py - ROS 2 Launch File for NAO Robot in Webots

This launch file starts the Webots simulator with the NAO robot
and attaches the ROS 2 controller (NaoDriver) for animation control.

What it does:
    1. Starts Webots with the nao_world.wbt simulation
    2. Loads the robot description from nao.urdf
    3. Attaches the NaoDriver controller to the NAO robot
    4. Registers shutdown handler for clean exit

Usage:
    ros2 launch my_nao_controller robot_launch.py

Requirements:
    - Webots R2025a installed
    - ros-humble-webots-ros2-driver package
    - WEBOTS_HOME environment variable set

Author: Vashu Chauhan
"""

import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    """
    Generate the ROS 2 launch description for the NAO simulation.
    
    Creates and configures:
        - WebotsLauncher: Starts Webots with the world file
        - WebotsController: Attaches the Python driver to the robot
        - Shutdown handler: Exits cleanly when Webots closes
    
    Returns:
        LaunchDescription: The complete launch configuration
    """
    # Get the installed package directory
    package_dir = get_package_share_directory('my_nao_controller')
    
    # Path to the world file
    world = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'nao_world.wbt')
    )

    # Path to the URDF file
    robot_description_path = os.path.join(package_dir, 'resource', 'nao.urdf')

    # The Driver Node
    # This launches your Python class as the controller
    nao_driver = WebotsController(
        robot_name='NAO', # Must match the name in the .wbt file
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    return LaunchDescription([
        world,
        nao_driver,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=world,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])