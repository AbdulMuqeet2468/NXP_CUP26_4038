# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
from pyzbar import pyzbar



class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic.
    """
    def __init__(self):
        super().__init__('qr_detector')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        self.last_qr = None
        self.stable_count = 0
        self.required_hits = 3
        self.published_qr = None

        self.get_logger().info("QR Detector Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        qr_data = self.detect_qr_code(image)

        if qr_data is None:
            self.last_qr = None
            self.stable_count = 0
            self.published_qr = None
            return

        if qr_data == self.last_qr:
            self.stable_count += 1
        else:
            self.last_qr = qr_data
            self.stable_count = 1

        if self.stable_count >= self.required_hits and qr_data != self.published_qr:
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.published_qr = qr_data
            
        self.get_logger().info(f"QR Data: {qr_data}")

    def detect_qr_code(self, image):
        """
        Detect and decode QR code in the image.
        """
        # Original image
        decoded = pyzbar.decode(image)
        if decoded:
            return decoded[0].data.decode("utf-8")

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)
        if decoded:
            return decoded[0].data.decode("utf-8")

        # Adaptive Threshold
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        decoded = pyzbar.decode(thresh)
        if decoded:
            return decoded[0].data.decode("utf-8")

        return None

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()