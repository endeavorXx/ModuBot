"""
op2_driver.py - Animation Engine for Robotis OP2 Robot in Webots

This module implements the Webots controller that receives action commands
via ROS 2 topics and translates them into smooth robot joint movements.
It supports both static poses (instant positions) and complex animations
(time-based sine wave interpolations).

The driver subscribes to the '/perform_action' topic and executes the
corresponding animation from the action vocabulary.
"""

import rclpy
import math
from std_msgs.msg import String
from webots_ros2_driver.webots_controller import WebotsController
from op2_controller.op2_action_vocab import OP2_ACTIONS, OP2_COMPLEX_ACTIONS

class Op2Driver(WebotsController):
    """
    Webots controller class for animating the Robotis OP2 robot.
    
    This class extends WebotsController to handle ROS 2 communication
    and execute robot animations. It manages motor positions for all
    joints and supports two types of actions:
    
    1. Static Actions: Instant pose changes (from OP2_ACTIONS)
    2. Complex Actions: Time-based animations using sine wave 
       interpolation (from OP2_COMPLEX_ACTIONS)
    
    Attributes:
        motors (dict): Dictionary mapping joint names to Webots motor devices
        current_animation (dict): Currently playing animation config, or None
        anim_start_time (float): Simulation time when current animation started
        ros_node: ROS 2 node for logging and subscriptions
    """
    
    def init(self, webots_node, properties):
        """
        Initialize the OP2 driver controller.
        
        Sets up ROS 2 communication, loads all motor devices from Webots,
        and prepares the animation system.
        
        Args:
            webots_node: The Webots node providing access to the robot
            properties: Configuration properties (unused)
        """
        # Ensure ROS 2 is initialized (required for node creation)
        if not rclpy.ok():
            rclpy.init(args=None)

        self.__robot = webots_node.robot
        self.ros_node = rclpy.create_node('op2_driver_node')
        self.ros_node.create_subscription(String, '/perform_action', self.action_callback, 100)
        
        # Load ALL Motors
        self.motors = {}
        all_joints = set()
        
        # Collect from Static
        for action in OP2_ACTIONS.values():
            all_joints.update(action.keys())
        
        # Collect from Complex
        for action in OP2_COMPLEX_ACTIONS.values():
            for curve in action["curves"]:
                all_joints.add(curve["joint"])
            
        for joint_name in all_joints:
            device = self.__robot.getDevice(joint_name)
            if device:
                self.motors[joint_name] = device
            else:
                self.ros_node.get_logger().warn(f"Motor not found: {joint_name}")
        
        self.current_animation = None
        self.anim_start_time = 0.0
        self.ros_node.get_logger().info("OP2 Driver Ready (Action Vocabulary with Descriptions)")

    def action_callback(self, msg):
        """
        ROS 2 callback for handling incoming action commands.
        
        Determines if the requested action is static or complex,
        then executes it accordingly.
        
        Args:
            msg (std_msgs.msg.String): Message containing the action name
        """
        action_name = msg.data
        self.ros_node.get_logger().info(f"Received Action: {action_name}")

        if action_name in OP2_COMPLEX_ACTIONS:
            self.start_animation(action_name)
        elif action_name in OP2_ACTIONS:
            self.current_animation = None
            for joint, value in OP2_ACTIONS[action_name].items():
                if joint in self.motors:
                    self.motors[joint].setPosition(value)
        else:
            self.ros_node.get_logger().warn(f"Unknown action: {action_name}")

    def start_animation(self, name):
        """
        Start a complex (time-based) animation.
        
        Loads the animation configuration, pre-calculates mathematical
        parameters for sine wave interpolation, and records the start time.
        
        Args:
            name (str): The name of the complex action to start
        
        Note:
            Animation parameters calculated:
            - center: Midpoint between min and max joint positions
            - amplitude: Half the range of motion
            - frequency: How fast to oscillate (repetitions / duration)
        """
        config = OP2_COMPLEX_ACTIONS[name]
        # Pre-calculate center point and amplitude for sine wave
        for curve in config["curves"]:
            curve["center"] = (curve["max"] + curve["min"]) / 2.0
            curve["amplitude"] = (curve["max"] - curve["min"]) / 2.0
        
        config["frequency"] = config["repetitions"] / config["duration"]
        self.current_animation = config
        self.anim_start_time = self.__robot.getTime()
        
        # Log the description (Good for debugging)
        if "description" in config:
            self.ros_node.get_logger().info(f"Performing: {config['description']}")

    def step(self):
        """
        Called every simulation timestep by Webots.
        
        This method:
        1. Processes any pending ROS 2 messages
        2. Updates motor positions if an animation is playing
        
        The animation uses sine wave interpolation to create smooth,
        natural-looking movements. Each joint follows its own curve
        defined by min/max positions and optional phase shift.
        """
        # Process ROS 2 callbacks without blocking
        if hasattr(self, 'ros_node') and self.ros_node:
            rclpy.spin_once(self.ros_node, timeout_sec=0)

        # Update animation if one is currently playing
        if self.current_animation:
            current_time = self.__robot.getTime()
            elapsed = current_time - self.anim_start_time
            
            if elapsed > self.current_animation["duration"]:
                self.current_animation = None
                return

            for curve in self.current_animation["curves"]:
                phase_shift = -math.pi / 2
                if curve.get("start_from_max", False):
                    phase_shift = math.pi / 2

                angle = curve["center"] + \
                        curve["amplitude"] * \
                        math.sin(2 * math.pi * self.current_animation["frequency"] * elapsed + phase_shift)
                
                joint_name = curve["joint"]
                if joint_name in self.motors:
                    self.motors[joint_name].setPosition(angle)

def main(args=None):
    """
    Entry point for standalone execution.
    """
    rclpy.init(args=args)
    rclpy.shutdown()
