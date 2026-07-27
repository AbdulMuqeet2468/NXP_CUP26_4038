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
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class ObjectRecognizer(Node):
    """
    ROS 2 node that reads compressed camera images and recognizes traffic signs
    using a TorchScript-exported YOLO model.

    Publishes detected labels on:
        /sign_board_detection
    """

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

        # Must match the class order used during training.
        self.class_names = [
            "A",
            "B",
            "C",
            "Left",
            "Right",
            "Straight",
            "X",
            "Y",
            "Z",
        ]

        self.conf_thres = 0.35
        self.input_size = 512  # same size used in training
        self.model = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            "best.torchscript",
        )

        try:
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model not found at: {model_path}")
            else:
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                self.get_logger().info(f"Loaded TorchScript model from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load TorchScript model: {e}")

        # Simple stability filter so the node does not spam noisy frame-by-frame labels.
        self.last_label = None
        self.stable_count = 0
        self.published_label = None
        self.required_hits = 3

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        sign_detected = self.classify_sign(image)

        if sign_detected is None:
            self.last_label = None
            self.stable_count = 0
            return

        if sign_detected == self.last_label:
            self.stable_count += 1
        else:
            self.last_label = sign_detected
            self.stable_count = 1

        if self.stable_count >= self.required_hits and sign_detected != self.published_label:
            msg = String()
            msg.data = sign_detected
            self.publisher_sign.publish(msg)
            self.published_label = sign_detected
            self.get_logger().info(f"Detected Sign Board: {sign_detected}")

    def letterbox(self, image, new_shape=512, color=(114, 114, 114)):
        """
        Resize image to a square while keeping aspect ratio, then pad.
        """
        shape = image.shape[:2]  # (h, w)

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        return image, r, (dw, dh)

    def _normalize_output(self, output):
        """
        Make TorchScript output easier to handle.

        Handles common cases:
        - tensor [1, N, 4+nc]
        - tensor [1, 4+nc, N]
        - tensor [N, 6]
        - list/tuple with one tensor
        """
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                return None
            output = output[0]

        if isinstance(output, dict):
            # If the model returns a dict, take the first tensor-like value.
            for v in output.values():
                if torch.is_tensor(v):
                    output = v
                    break

        if not torch.is_tensor(output):
            return None

        if output.ndim == 3:
            output = output[0]
            if output.shape[0] < output.shape[1]:
                output = output.permute(1, 0)

        return output

    def classify_sign(self, image):
        if self.model is None:
            return None

        try:
            img, _, _ = self.letterbox(image, self.input_size)

            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # HWC -> CHW, normalize to 0..1
            img = torch.from_numpy(img).to(self.device)
            img = img.permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0)

            with torch.no_grad():
                output = self.model(img)

            pred = self._normalize_output(output)
            if pred is None:
                return None

            # Case 1: YOLO-style output with class scores
            # Shape like [N, 4 + nc]
            if pred.ndim == 2 and pred.shape[1] >= 4 + len(self.class_names):
                class_scores = pred[:, 4:4 + len(self.class_names)]
                best_scores, best_classes = torch.max(class_scores, dim=1)

                best_idx = torch.argmax(best_scores)
                confidence = float(best_scores[best_idx].item())
                class_id = int(best_classes[best_idx].item())

                if confidence < self.conf_thres:
                    return None

                return self.class_names[class_id]

            # Case 2: already post-processed output like [N, 6]
            # Expected: x1, y1, x2, y2, conf, cls
            if pred.ndim == 2 and pred.shape[1] >= 6:
                confidences = pred[:, 4]
                best_idx = torch.argmax(confidences)

                confidence = float(confidences[best_idx].item())
                if confidence < self.conf_thres:
                    return None

                class_id = int(pred[best_idx, 5].item())
                if 0 <= class_id < len(self.class_names):
                    return self.class_names[class_id]

                return None

            return None

        except Exception as e:
            self.get_logger().debug(f"Inference failed: {e}")
            return None


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