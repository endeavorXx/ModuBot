"""
test_op2_actions.py - Manual Action Testing Tool

This script provides a simple way to test OP2 robot actions without
running the full AI pipeline. It publishes a sequence of predefined
action commands to the /perform_action topic.

Useful for:
    - Verifying robot animations work correctly
    - Testing new actions after adding them to the vocabulary
    - Debugging driver issues in isolation

Usage:
    Terminal 1: ros2 launch op2_controller robot_launch.py
    Terminal 2: ros2 run op2_controller test_actions
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class Op2ActionTester(Node):
    """
    ROS 2 node for testing robot actions sequentially.
    
    Publishes a predefined list of actions to /perform_action topic
    with appropriate delays to allow animations to complete.
    """
    
    def __init__(self):
        """
        Initialize the tester node with a list of actions to perform.
        """
        super().__init__('op2_action_tester')
        self.publisher_ = self.create_publisher(String, '/perform_action', 10)
        # List of actions to test: (action_name, wait_duration_seconds)
        self.actions = [
            ("stand_neutral", 3),
            ("wave_right_hand", 4),
            ("nod_head", 3),
            ("shake_head", 3),
            ("stand_neutral", 2)
        ]
        self.timer = self.create_timer(1.0, self.run_test)
        self.current_idx = 0
        self.get_logger().info("Starting OP2 Action Test Sequence...")

    def run_test(self):
        """
        Execute the next action in the sequence.
        """
        if self.current_idx < len(self.actions):
            action, duration = self.actions[self.current_idx]
            msg = String()
            msg.data = action
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published: {action} (Waiting {duration}s)")
            
            # Update timer for next action
            self.timer.cancel()
            self.timer = self.create_timer(duration, self.run_test)
            self.current_idx += 1
        else:
            self.get_logger().info("Test Sequence Complete.")
            self.timer.cancel()
            raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    tester = Op2ActionTester()
    try:
        rclpy.spin(tester)
    except SystemExit:
        pass
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
