import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TargetPublisher(Node):

    def __init__(self):
        super().__init__("target_publisher")

        self.pub = self.create_publisher(
            String,
            "/target_patient",
            10
        )

        self.timer = self.create_timer(1.0, self.publish_target)

    def publish_target(self):
        msg = String()
        msg.data = "B"      # Change to A, C, X, Y, Z for testing

        self.pub.publish(msg)
        self.get_logger().info(f"Published target: {msg.data}")


def main(args=None):
    rclpy.init(args=args)

    node = TargetPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()