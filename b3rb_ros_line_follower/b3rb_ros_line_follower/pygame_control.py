#!/usr/bin/env python3

import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

FORWARD_SPEED = 2.0#0.685
REVERSE_SPEED = -2.0
LEFT_TURN = 0.45
RIGHT_TURN = -0.45


class JoyTeleop(Node):

    def __init__(self):
        super().__init__("joy_teleop")

        self.pub = self.create_publisher(
            Joy,
            "/cerebri/in/joy",
            10
        )

        self.speed = 0.0
        self.turn = 0.0

        self.timer = self.create_timer(0.02, self.publish_cmd)   # 50 Hz

    def publish_cmd(self):

        msg = Joy()

        msg.axes = [
            0.0,
            self.speed,
            0.0,
            self.turn
        ]

        msg.buttons = [1,0,0,0,0,0,0,1]

        self.pub.publish(msg)


def main():

    rclpy.init()

    node = JoyTeleop()

    pygame.init()

    screen = pygame.display.set_mode((300,150))
    pygame.display.set_caption("NXP Joy Teleop")

    clock = pygame.time.Clock()

    running = True

    while running and rclpy.ok():

        rclpy.spin_once(node, timeout_sec=0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Default
        node.speed = 0.0
        node.turn = 0.0

        # Speed
        if keys[pygame.K_w]:
            node.speed = FORWARD_SPEED

        elif keys[pygame.K_s]:
            node.speed = REVERSE_SPEED

        # Steering
        if keys[pygame.K_a]:
            node.turn = LEFT_TURN

        elif keys[pygame.K_d]:
            node.turn = RIGHT_TURN

        screen.fill((30,30,30))

        pygame.display.set_caption(
            f"Speed={node.speed:.2f}   Turn={node.turn:.2f}"
        )

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()