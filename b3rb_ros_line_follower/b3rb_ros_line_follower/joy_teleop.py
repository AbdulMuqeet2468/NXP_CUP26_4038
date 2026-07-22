#!/usr/bin/env python3

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


class JoyTeleop(Node):

    def __init__(self):
        super().__init__('joy_teleop')

        self.publisher = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            10
        )

        self.speed = 0.0
        self.turn = 0.0

        self.timer = self.create_timer(0.05, self.publish_cmd)   # 20 Hz

        self.get_logger().info("=== Joy Teleop ===")
        self.get_logger().info("w/s : Forward / Reverse")
        self.get_logger().info("a/d : Left / Right")
        self.get_logger().info("space : Stop")
        self.get_logger().info("q : Quit")

    def publish_cmd(self):
        msg = Joy()

        msg.axes = [
            0.0,
            self.speed,
            0.0,
            self.turn
        ]

        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]

        self.publisher.publish(msg)


def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():

    rclpy.init()

    node = JoyTeleop()

    try:
        while rclpy.ok():

            rclpy.spin_once(node, timeout_sec=0)

            key = get_key()

            if key == 'w':
                node.speed = min(node.speed + 0.05, 1.0)

            elif key == 's':
                node.speed = max(node.speed - 0.05, -1.0)

            elif key == 'a':
                node.turn = min(node.turn + 0.1, 1.0)

            elif key == 'd':
                node.turn = max(node.turn - 0.1, -1.0)

            elif key == ' ':
                node.speed = 0.0
                node.turn = 0.0

            elif key == 'q':
                break

            print(
                f"\rSpeed: {node.speed:.2f}   Turn: {node.turn:.2f}",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()