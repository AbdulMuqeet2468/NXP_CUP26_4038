import rclpy
from rclpy.node import Node
import time
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication
import numpy as np

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0


class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    By default, it publishes a safe drive-straight command on a timer loop.
    Implement logic inside the callbacks to steer, dodge obstacles, detect destinations,
    communicate with the server, and park.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        print("RUNNER VERSION 2")
        # 1. Lane Edge Vectors (from edge_vectors_publisher)
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # 2. LIDAR Obstacle Scanner
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # 3. Server Communication Feedback Loop
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # 4. QR Code Detections (from qr_detector)
        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        # 5. Sign Board Detections (from object_recognizer)
        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        
        # Publisher to drive/steer the buggy
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher to send messages to the Server
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # Publisher to send target patient info to the object recognizer
        self.publisher_target = self.create_publisher(
                String,
                "/target_patient",
                10
            )

        # ------------------ State Variables & Timer ------------------
        
        # Default controls
        self.target_speed = 0.0
        self.target_turn = 0.0

        # State variables (You can add your own state flags / state machines here)
        self.obstacle_in_front = False
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = None
        self.mission_completed = False

        #added
        
        # earlier good values
        #self.kp = 0.0076
        #self.kd = 0.004
        #self.ki = 0.0000125

        # Tuned PID values for 'good' performance
        # currently used for both single and double lane detection
        self.kp = 0.00775
        self.kd = 0.00425
        self.ki = 0.000014

        # Separate integral/prev_error state per control regime, so switching
        # between CENTER (one/two lane), STICK_LEFT, and STICK_RIGHT doesn't
        # carry over error history from a different regime.
        self.pid_state = {
            "CENTER_TWO_LANE": {"integral": 0.0, "prev_error": 0.0},
            "CENTER_ONE_LANE": {"integral": 0.0, "prev_error": 0.0},
            "STICK_LEFT":      {"integral": 0.0, "prev_error": 0.0},
            "STICK_RIGHT":     {"integral": 0.0, "prev_error": 0.0},
        }

        # LIDAR flags
        self.obj_on_right = False
        self.obj_on_left = False
        self.obj_still_on_right = False
        self.obj_still_on_left = False

        # LIDAR distance thresholds 
        self.min_threshold = 0.5
        self.detection_threshold = 1.5 #1.5

        # Lane
        self.lane_mode = "CENTER"  # Options: "CENTER", "STICK_LEFT", "STICK_RIGHT", "TURN_LEFT", "TURN_RIGHT"

        # Flags for turn execution
        self.sign_detected = False
        self.pending_turn = None
        self.turn_now = False
        self.lane_lost_during_turn = False

        # Turn timing variables
        self.turn_start_time = 0.0
        self.min_turn_time = 2.5     # seconds

        # Have to change this
        self.targets = ["A", "X", "B", "Y", "C", "Z"]
        self.current_target = 0

        self.set_target_patient(self.targets[self.current_target])

        # turning logic parameters (under review)
        self.lane_lost_during_turn = False
        self.lane_lost_start_time = 0.0

        self.turn_step_base = 0.00085
        self.turn_step_max = 0.004
        self.turn_escalate_after = 2.5   # seconds before escalating
        self.turn_escalate_rate = 0.0065  # extra step added per second past that

        # Timer to publish drive commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        # Timer to debug status
        self.create_timer(1.0, self.debug_status)

        self.get_logger().info("Line Follower controller initialized. Starting Mission.")

    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    def _pid_step(self, key, error):
        """
        Run one PID step using the integral/prev_error belonging ONLY to
        `key` (e.g. "CENTER_TWO_LANE", "STICK_LEFT", ...), so different
        control regimes never share or mix accumulated error state.
        """
        state = self.pid_state[key]
        state["integral"] += error
        state["integral"] = max(min(state["integral"], 1000), -1000)
        derivative = error - state["prev_error"]
        state["prev_error"] = error
        turn = self.kp * error + self.ki * state["integral"] + self.kd * derivative
        return max(min(turn, 1.0), -1.0)

    def _reset_pid_state(self, key=None):
        """Reset one regime's PID state, or all of them if key is None."""
        if key is None:
            for k in self.pid_state:
                self.pid_state[k] = {"integral": 0.0, "prev_error": 0.0}
        else:
            self.pid_state[key] = {"integral": 0.0, "prev_error": 0.0}

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        """
        Receives lane boundaries from the camera vector extractor
        """
        
        self.edge_vector_count = message.vector_count

        if self.turn_now:
            self.execute_turn()
            return

        width = message.image_width
        image_center = width / 2.0

        # No lane detected
        if message.vector_count == 0:
            self.target_turn = 0.0
            return

        # Compute lane geometry once (same as CENTER's two-lane case) so
        # STICK_LEFT and STICK_RIGHT can both target a point relative to the
        # actual lane, exactly like CENTER does, instead of one side using a
        # dynamic lane-relative target and the other a hardcoded fixed pixel
        # value (that asymmetry was the actual bug).
        two_lanes = (message.vector_count == 2)
        if two_lanes:
            left_x = message.vector_1[1].x
            right_x = message.vector_2[1].x
            lane_width = right_x - left_x
            margin = 0.65 * lane_width
            lane_center = (left_x + right_x) / 2.0

        if self.lane_mode == "CENTER":
            # ------------------ Two lanes ------------------
            if two_lanes:

                self.target_speed = 1.0

                # PID control for two lane control
                #self.kp = 0.00775
                #self.kd = 0.00425
                #self.ki = 0.000014

                # PID control for lane following    
                error = lane_center - image_center
                self.target_turn = self._pid_step("CENTER_TWO_LANE", error)
                return
       
            # ------------------ One lane ------------------
            self.target_speed = 0.685

            #trying values for single lane control
            #self.kp = 0.00775
            #self.kd = 0.00425
            #self.ki = 0.000014

            x = message.vector_1[1].x

            # Line on left side of image
            if x < image_center:
                desired_x = width * 0.15

            # Line on right side
            else:
                desired_x = width * 0.85

            error = x - desired_x
            self.target_turn = self._pid_step("CENTER_ONE_LANE", error)
        
        elif self.lane_mode == "STICK_LEFT":

            if self.turn_now:
                return
            self.target_speed = 0.5

            x = message.vector_1[1].x

            if two_lanes:
                # Lane-relative target, mirroring STICK_RIGHT and CENTER:
                # aim for a point `margin` to the left of the actual lane
                # center, using the real detected lane geometry.
                desired_x = lane_center - margin
            else:
                # Only one lane visible - no lane_center to offset from, so
                # fall back to a fixed pixel target (same approximation
                # CENTER's one-lane case already uses).
                desired_x = width * 0.15

            error = x - desired_x

            self.target_turn = self._pid_step("STICK_LEFT", error)

        else: # self.lane_mode == "STICK_RIGHT"
        
            if self.turn_now:
                return

            self.target_speed = 0.5

            if two_lanes:
                x = message.vector_2[1].x
                desired_right = lane_center + margin
            else:
                # Only one lane visible - lane_center/margin don't exist this
                # cycle, so fall back to a fixed pixel target instead of
                # referencing undefined variables (this crashed before).
                x = message.vector_1[1].x
                desired_right = width * 0.85

            error = x - desired_right
            self.target_turn = self._pid_step("STICK_RIGHT", error)

    def lidar_callback(self, message):
        #return


        if self.turn_now and not self.sign_detected:
            return
        
        # LIDAR sectors
        front_left = message.ranges[195:239]
        front_right = message.ranges[120:165]
        left = message.ranges[240:300]
        right = message.ranges[60:119]

        front_right_vals = [x for x in front_right if np.isfinite(x)]
        front_left_vals = [x for x in front_left if np.isfinite(x)]
        right_vals = [x for x in right if np.isfinite(x)]
        left_vals = [x for x in left if np.isfinite(x)]

        front_right_min = min(front_right_vals) if front_right_vals else float('inf')
        front_left_min = min(front_left_vals) if front_left_vals else float('inf')
        right_avg = np.mean(right_vals) if right_vals else float("inf")
        left_avg = np.mean(left_vals) if left_vals else float("inf")

        # Estimate direction of objects
        if front_right_min < self.detection_threshold:
            self.obj_on_right = True
        else:
            self.obj_on_right = False

        if front_left_min < self.detection_threshold:
            self.obj_on_left = True
        else:
            self.obj_on_left = False

        # Check if objects are still present in the same direction
        #Check right side
        if right_avg < self.min_threshold:
            self.obj_still_on_right = True
        else:
            self.obj_still_on_right = False
        
        #Check left side
        if left_avg < self.min_threshold:
            self.obj_still_on_left = True
        else:
            self.obj_still_on_left = False

        # Based on side checks, check if object is still present in the same direction
        if self.obj_on_right:
            self.lane_mode = "STICK_LEFT"

        elif self.obj_on_left:
            self.lane_mode = "STICK_RIGHT"

        elif self.obj_still_on_right:
            self.lane_mode = "STICK_LEFT"

        elif self.obj_still_on_left:
            self.lane_mode = "STICK_RIGHT"

        else:
            self.lane_mode = "CENTER"
            
    def server_communication_callback(self, message):
        """
        Receives coordination commands from the server.
        
        GUIDELINE (Server Communication):
        - Check if the message is destined for the Buggy (`message.dest == 1`).
		- Do not forget to check for ACK messages from server
        - The server communicates mission info in the `message.msg` payload string.
        - Parse server instructions (e.g., patient pickup, target hospitals).
        - Call `self.send_server_update` to report your status when you reach a checkpoint.
        """
        if message.dest == 1:
            self.get_logger().info(f"Received Server Message: {message.msg}")
            # Parse payload and update state machine destination/objectives here
            pass

    def send_server_update(self, text_msg):
        """Sends status messages to the server. (Do not forget to send ACK messages to server)"""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = 100     # Replace with a rolling message ID/counter
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.
        
        GUIDELINE (Patient/Hospital Identification):
        - Parse the decoded string payload in `message.data` (e.g. "PATIENT_A", "HOSPITAL_B").
        - If it matches your target destination, stop the vehicle close to the building (verify range using LIDAR),
          perform the action (pick patient / drop patient), and communicate the arrival to the server.
        """
        #self.next_target()
        self.get_logger().info(f"Heard QR code: {message.data}")
        pass

    def sign_board_callback(self, message):


        if message.data == "CLEAR":

            self.sign_detected = False

            if self.pending_turn is not None:
                self.turn_now = True
                self.lane_lost_during_turn = False 
                self.lane_lost_start_time = 0.0

            return

        self.sign_detected = True
        self.pending_turn = message.data

        
    def debug_status(self):
        self.get_logger().info(
            f"Mode={self.lane_mode}"
        )

        self.get_logger().info(
            f"Pending={self.pending_turn} | "
            f"Sign={self.sign_detected} | "
            f"TurnNow={self.turn_now}"
        )

    def set_target_patient(self, patient):

        msg = String()
        msg.data = patient

        self.publisher_target.publish(msg)

        self.get_logger().info(f"Target patient: {patient}")

    def next_target(self):

        if self.current_target < len(self.targets) - 1:
            self.current_target += 1
            self.set_target_patient(self.targets[self.current_target])
        else:
            self.get_logger().info("Mission Complete!")

    def execute_turn(self):

        self.target_speed = 0.75

        if self.edge_vector_count == 0:
            if not self.lane_lost_during_turn:
                self.lane_lost_during_turn = True
                self.lane_lost_start_time = time.time()
                self.get_logger().info(
                    f"[TURN] Lane lost. pending={self.pending_turn} | "
                    f"starting escalation timer"
                )

        if not self.lane_lost_during_turn or self.edge_vector_count == 0:

            lost_duration = 0.0
            if self.lane_lost_start_time > 0.0:
                lost_duration = time.time() - self.lane_lost_start_time

            step_mag = self.turn_step_base
            if lost_duration > self.turn_escalate_after:
                step_mag += (lost_duration - self.turn_escalate_after) * self.turn_escalate_rate
                step_mag = min(step_mag, self.turn_step_max)

            step = 0.0
            if self.pending_turn == "Right":
                step = -step_mag
            elif self.pending_turn == "Left":
                step = step_mag

            self.target_turn += step
            self.target_turn = max(min(self.target_turn, 1.0), -1.0)

            self.get_logger().info(
                f"[TURN] pending={self.pending_turn} | "
                f"edge_count={self.edge_vector_count} | "
                f"lost_duration={lost_duration:.2f}s | "
                f"step_mag={step_mag:.5f} | "
                f"target_turn={self.target_turn:.4f}"
            )

        else:
            self.get_logger().info(
                f"[TURN] Lane reacquired. pending={self.pending_turn} | "
                f"turn complete, resetting state"
            )
            # lane was lost earlier AND is now found again -> turn succeeded
            self.target_turn = 0.0
            self.turn_now = False
            self.pending_turn = None
            self.lane_lost_during_turn = False
            self.lane_lost_start_time = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()