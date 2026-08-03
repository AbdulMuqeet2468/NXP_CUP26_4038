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

import os

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory
package_share = get_package_share_directory("b3rb_ros_line_follower")


class ObjectRecognizer(Node):

    def __init__(self):
        super().__init__("object_recognizer")

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.camera_image_callback,
            10,
        )

        self.publisher_sign = self.create_publisher(
            String,
            "/sign_board_detection",
            10,
        )

        
        model_path = os.path.join(package_share, "best.pt")

        self.model = YOLO(model_path)

        self.target_patient = None
        self.last_label = None
        self.stable_count = 0
        self.published_label = None
        self.required_hits = 3

        self.sign_visible = False

        self.subscription_target = self.create_subscription(
            String,
            "/target_patient",
            self.target_callback,
            10
        )

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        sign_detected = self.classify_sign(image)

        if sign_detected is None:

            if self.sign_visible:
                msg = String()
                msg.data = "CLEAR"
                self.publisher_sign.publish(msg)

            self.sign_visible = False
            self.last_label = None
            self.stable_count = 0
            self.published_label = None
            return
        #problem here is that whenever sign board is out of camera fov and again the same board comes in fov, it is detecting again, and mightrepeat the same command
        

        if sign_detected == self.last_label:
            self.stable_count += 1
        else:
            self.last_label = sign_detected
            self.stable_count = 1

        if self.stable_count >= self.required_hits and sign_detected != self.published_label:
            msg = String()
            msg.data = sign_detected
            self.publisher_sign.publish(msg)
            self.sign_visible = True
            self.published_label = sign_detected
            self.get_logger().info(f"Detected Sign Board: {sign_detected}")

    def target_callback(self, msg):
        self.target_patient = msg.data

        self.last_label = None
        self.stable_count = 0
        self.published_label = None
        self.sign_visible = False

        self.get_logger().info(f"Looking for patient: {self.target_patient}")
    
    def classify_sign(self, image):

        if self.target_patient is None:
            return None

        results = self.model(image, verbose=False, conf=0.40)

        boxes = results[0].boxes

        if len(boxes) == 0:
            return None

        patient_box = None
        direction_boxes = []

        # Collect detections
        for box in boxes:

            cls = int(box.cls.item())
            label = self.model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detection = {
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2
            }

            if label == self.target_patient:
                patient_box = detection

            elif label in ["Left", "Right", "Straight"]:
                direction_boxes.append(detection)

        if patient_box is None:
            return None

        best_direction = None
        best_distance = float("inf")

        for direction in direction_boxes:

            # Direction must be below the patient
            if direction["cy"] <= patient_box["cy"]:
                continue

            # Direction should be horizontally aligned
            if abs(direction["cx"] - patient_box["cx"]) > patient_box["x2"] - patient_box["x1"]:
                continue

            distance = direction["cy"] - patient_box["cy"]

            if distance < best_distance:
                best_distance = distance
                best_direction = direction["label"]

        return best_direction
       

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()