"""
test_nao_actions.py - Manual Action Testing Tool

This script provides a simple way to test NAO robot actions without
running the full AI pipeline. It publishes a sequence of predefined
action commands to the /perform_action topic.

Useful for:
    - Verifying robot animations work correctly
    - Testing new actions after adding them to the vocabulary
    - Debugging driver issues in isolation

Usage:
    Terminal 1: ros2 launch my_nao_controller robot_launch.py
    Terminal 2: ros2 run my_nao_controller test_actions

Author: Vashu Chauhan
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class NaoActionTester(Node):
    """
    ROS 2 node for testing robot actions sequentially.
    
    Publishes a predefined list of actions to /perform_action topic
    with appropriate delays to allow animations to complete.
    
    Attributes:
        publisher_: ROS 2 publisher for action commands
        actions (list): Tuples of (action_name, wait_duration)
    """
    
    def __init__(self):
        """
        Initialize the tester node with a list of actions to perform.
        
        Modify the 'actions' list to test different action sequences.
        Each tuple contains (action_name, wait_seconds).
        """
        super().__init__('nao_action_tester')
        self.publisher_ = self.create_publisher(String, '/perform_action', 10)
        # List of actions to test: (action_name, wait_duration_seconds)
        # Wait duration should be >= action duration to see full animation
        self.actions = [
            # ("stand_neutral", 2), # Action Name, Sleep Duration
            # ("shake_head", 4),    # Duration is 3s, we wait 4s to be safe
            # ("stand_neutral", 2),
            # ("nod_head", 4),      # Duration is 3s, we wait 4s
            # ("wave_hand", 3),
            # ("stand_neutral", 1)
            ("stand_neutral", 2),
            ("get_out_left_hand", 5), # Action takes 4s, we wait 5s
            ("stand_neutral", 2),
            ("get_out_right_hand", 5), # Action takes 4s, we wait 5s
            ("stand_neutral", 2)
        ]

    def run_test(self):
        """
        Execute the action test sequence.
        
        Publishes each action in the list, then waits the specified
        duration before sending the next action.
        """
        print("Starting Complex Action Test...")
        time.sleep(2) # Wait for ROS connection to establish
        
        for action, duration in self.actions:
            msg = String()
            msg.data = action
            print(f"Sending Action: {action} (Waiting {duration}s)")
            self.publisher_.publish(msg)
            
            # Wait for robot to complete the animation
            time.sleep(duration) 

def main(args=None):
    """
    Entry point for the action tester.
    
    Initializes ROS 2, runs the test sequence, then shuts down.
    
    Args:
        args: Command-line arguments (passed to rclpy.init)
    """
    rclpy.init(args=args)
    tester = NaoActionTester()
    tester.run_test()
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()