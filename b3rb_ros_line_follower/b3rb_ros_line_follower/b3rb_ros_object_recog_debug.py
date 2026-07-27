import os

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ament_index_python.packages import get_package_share_directory
package_share = get_package_share_directory("b3rb_ros_line_follower")



class ObjectRecognizerDebug(Node):

    def __init__(self):
        super().__init__("object_recognizer_debug")

        self.subscription = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.camera_callback,
            10
        )

        self.publisher_debug = self.create_publisher(
            CompressedImage,
            "/camera/image_detection/compressed",
            10
        )

        model_path = os.path.join(package_share, "best.pt")

        self.model = YOLO(model_path)

        self.get_logger().info("Loaded YOLO model.")
        self.get_logger().info("Debug node started.")

    def camera_callback(self, message):

        # Decode ROS compressed image
        np_arr = np.frombuffer(message.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # Run YOLO
        results = self.model(
            frame,
            verbose=False,
            conf=0.40
        )

        # Draw detections
        annotated = results[0].plot()

        # Convert back to ROS CompressedImage
        debug_msg = CompressedImage()
        debug_msg.header = message.header
        debug_msg.format = "jpeg"

        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            return

        debug_msg.data = encoded.tobytes()

        # Publish
        self.publisher_debug.publish(debug_msg)

def main(args=None):

    rclpy.init(args=args)

    node = ObjectRecognizerDebug()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()