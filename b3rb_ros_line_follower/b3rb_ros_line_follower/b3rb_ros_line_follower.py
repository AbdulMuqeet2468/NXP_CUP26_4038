import rclpy
from rclpy.node import Node
import time
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication
import numpy as np
import random

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
        self.building_on_right = False
        self.building_on_left = False

        # LIDAR distance thresholds 
        self.min_threshold = 0.5
        self.detection_threshold = 1.5 #1.5
        self.building_on_right_threshold = 3.0
        self.building_on_left_threshold = 3.0

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

        self.hospital_detected = False
        self.patient_detected = False

        # Sub-state for the new turn logic:
        # "STRAIGHT" -> drive straight until lane vanishes (vector_count == 0)
        # "TURNING"  -> apply fixed turn value until lane reappears (vector_count >= 1)
        self.turn_phase = "STRAIGHT"

        # Fixed turn magnitude used during the TURNING phase
        self.turn_fixed_value = 0.45

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

        # ---- Mission search state ----
        self.search_patient = True     # informational flag, mirrors mission_phase
        self.search_hospital = False   # informational flag, mirrors mission_phase

        # Detector format -> server format ("A", "X", etc. per protocol)
        self.qr_to_server_patient = {"PATIENT_1": "A", "PATIENT_2": "B", "PATIENT_3": "C"}
        self.qr_to_server_hospital = {"HOSPITAL_1": "X", "HOSPITAL_2": "Y", "HOSPITAL_3": "Z"}

        # Which QR payloads mean "patient" vs "hospital". Adjust these sets
        # if your actual codes differ.
        self.patient_codes = set(self.qr_to_server_patient.keys())
        self.hospital_codes = set(self.qr_to_server_hospital.keys())

        # ---- Mission phase state machine ----
        # SEARCHING_PATIENT          -> normal driving, looking for a patient QR
        # PATIENT_QR_READ_WAIT_ACK   -> sent patient id to server, waiting for ack
        # WAITING_HOSPITAL_LOCATION  -> acked; waiting for server to send hospital
        # SEARCHING_HOSPITAL         -> normal driving, looking for target hospital QR
        # HOSPITAL_QR_READ_WAIT_ACK  -> sent hospital-found to server, waiting for ack
        # WAITING_NEXT_PATIENT       -> acked; waiting for server to send next patient
        self.mission_phase = "SEARCHING_PATIENT"

        self.watching_for_stop = False  # True after QR disappears, until LIDAR says "inside premises"

        # Buggy must stay parked (stopped) in all of these phases - it's
        # mid-handshake with the server and shouldn't keep driving.
        self.waiting_phases = {
            "PATIENT_QR_READ_WAIT_ACK",
            "WAITING_HOSPITAL_LOCATION",
            "HOSPITAL_QR_READ_WAIT_ACK",
            "WAITING_NEXT_PATIENT",
        }

        # Tracks the message WE sent that we're waiting on an ack for.
        # {"uid": <int>, "purpose": "PATIENT_FOUND" | "HOSPITAL_FOUND"} or None
        self.pending_outgoing = None

        # ---- QR handling state ----
        # qr_state: "IDLE"    -> normal driving, just watching for a QR
        #           "READING" -> QR currently visible, drive straight
        #           "WAITING" -> QR just went out of view, stop + process it
        self.qr_state = "IDLE"
        self.qr_is_visible = False
        self.last_qr_value = None

        self.uid = None  # last message uid we sent to the server

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

        # While we're mid-handshake with the server (sent a message and
        # waiting for ack, or waiting for the server's next instruction),
        # stay completely parked. Don't drive, don't turn.
        if self.mission_phase in self.waiting_phases:
            self.target_speed = 0.0
            self.target_turn = 0.0
            return

        if self.turn_now:
            self.execute_turn()
            return

        # Continuously watch (every tick, not just once) for LIDAR signal
        # that we're inside the premises, if we're past a QR read.
        if self.watching_for_stop and (self.building_on_right or self.building_on_left):
            self.get_logger().info("[QR] Inside premises (LIDAR). Stopping to process.")
            self.qr_state = "WAITING"
            self.watching_for_stop = False

        # While a QR is being read or just finished being read, override
        # normal lane-following entirely.
        #if self.qr_state in ("READING", "WAITING"):
        if self.qr_state == "WAITING":
            self.handle_qr_state()
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
        narrow_left = message.ranges[295:315]
        narrow_right = message.ranges[55:65]

        front_right_vals = [x for x in front_right if np.isfinite(x)]
        front_left_vals = [x for x in front_left if np.isfinite(x)]
        right_vals = [x for x in right if np.isfinite(x)]
        left_vals = [x for x in left if np.isfinite(x)]
        narrow_left_vals = [x for x in narrow_left if np.isfinite(x)]
        narrow_right_vals = [x for x in narrow_right if np.isfinite(x)]

        front_right_min = min(front_right_vals) if front_right_vals else float('inf')
        front_left_min = min(front_left_vals) if front_left_vals else float('inf')
        right_avg = np.mean(right_vals) if right_vals else float("inf")
        left_avg = np.mean(left_vals) if left_vals else float("inf")
        narrow_left_avg = np.mean(narrow_left_vals) if narrow_left_vals else float("inf")
        narrow_right_avg = np.mean(narrow_right_vals) if narrow_right_vals else float("inf")

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

        # flags for building detection
        if narrow_right_avg < self.building_on_right_threshold:
            self.building_on_right = True
        else:
            self.building_on_right = False
        
        if narrow_left_avg < self.building_on_left_threshold:
            self.building_on_left = True    
        else:
            self.building_on_left = False

    def server_communication_callback(self, message):
        """
        Handles two kinds of incoming server messages:

        1. ack == 1  -> The server is acking something WE sent. Match the
           uid against self.pending_outgoing, then advance the mission
           phase depending on what that message's purpose was.

        2. ack == 0  -> The server is pushing NEW info to us (hospital
           location or next patient target). We must ack it back
           immediately, then act on the content depending on which
           phase we're currently waiting in.
        """
        if message.dest != 1:
            return  # not addressed to the buggy

        self.get_logger().info(
            f"[SERVER] Received src={message.src} uid={message.uid} "
            f"ack={message.ack} msg='{message.msg}'"
        )

        # ---------------- Case 1: server acking OUR message ----------------
        if message.ack == 1:
            if self.pending_outgoing is not None and message.uid == self.pending_outgoing["uid"]:
                purpose = self.pending_outgoing["purpose"]
                self.pending_outgoing = None

                self.get_logger().info(f"[SERVER] Ack confirmed uid={message.uid} purpose={purpose}")

                if purpose == "PATIENT_FOUND":
                    # Server confirmed receipt of patient id.
                    # Next: wait for server to send hospital location.
                    self.mission_phase = "WAITING_HOSPITAL_LOCATION"

                elif purpose == "HOSPITAL_FOUND":
                    # Server confirmed receipt of hospital-found.
                    # Next: wait for server to send the next patient target.
                    self.mission_phase = "WAITING_NEXT_PATIENT"
            else:
                self.get_logger().warn(
                    f"[SERVER] Got ack for uid={message.uid} but no matching pending message"
                )
            return

        # ---------------- Case 2: new message from server, needs our ack ----------------
        if message.ack == 0:
            # Ack it immediately, regardless of content, per protocol.
            self._publish_server_message(dest=2, uid=message.uid, ack=1, msg="")
            self.get_logger().info(f"[SERVER] Acked uid={message.uid}")

            if self.mission_phase == "WAITING_HOSPITAL_LOCATION":
                hospital = message.msg
                self.hospital_id = hospital
                self.current_destination = hospital

                self.search_patient = False
                self.search_hospital = True
                self.mission_phase = "SEARCHING_HOSPITAL"

                # Tell the detector node to now look for the hospital target,
                # not the patient. (Same topic - detector treats it as
                # "the thing to currently search for".)
                self.set_target_patient(hospital)

                self.get_logger().info(
                    f"[SERVER] Target hospital set to '{hospital}'. Resuming mission."
                )

            elif self.mission_phase == "WAITING_NEXT_PATIENT":
                patient = message.msg
                self.patient_id = None
                self.hospital_id = None
                self.current_destination = None

                self.search_patient = True
                self.search_hospital = False
                self.mission_phase = "SEARCHING_PATIENT"

                # Tell the object_recognizer node about the new target patient
                self.set_target_patient(patient)

                self.get_logger().info(
                    f"[SERVER] Next target patient: '{patient}'. Resuming mission."
                )

            else:
                self.get_logger().warn(
                    f"[SERVER] Unexpected unsolicited message '{message.msg}' "
                    f"while in phase {self.mission_phase}"
                )

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.

        ASSUMPTION: /qr_detection publishes continuously, sending an
        empty string (or "NONE") when no QR is currently in frame, and
        the decoded QR string when one IS in frame. If your qr_detector
        node instead only publishes when it sees something (i.e. never
        publishes "nothing visible"), you'll need a different way to
        detect the visible->not-visible transition (e.g. a timeout since
        the last message). Adjust `qr_now_visible` below accordingly.
        """
        data = message.data.strip()
        qr_now_visible = (data != "" and data != "NONE")

        if qr_now_visible:
            self.last_qr_value = data

        # Transition: visible -> not visible => resume normal driving,
        # but start watching LIDAR for "inside premises" signal.
        if (not qr_now_visible) and self.qr_is_visible and self.qr_state == "READING":
            self.get_logger().info("[QR] QR no longer visible. Resuming drive, watching LIDAR.")
            self.qr_state = "IDLE"
            self.watching_for_stop = True

        # Transition: not visible -> visible => start of a QR read
        if qr_now_visible and not self.qr_is_visible:
            self.get_logger().info(f"[QR] QR code detected: {data}")
            self.qr_state = "READING"

        self.qr_is_visible = qr_now_visible
        

    def sign_board_callback(self, message):


        if message.data == "CLEAR":

            self.sign_detected = False

            if self.pending_turn is not None:
                self.turn_now = True
                self.lane_lost_during_turn = False 
                self.lane_lost_start_time = 0.0

                # New turn is starting -> always begin in STRAIGHT phase
                self.turn_phase = "STRAIGHT"

            return

        self.sign_detected = True
        self.pending_turn = message.data

    def generate_uid(self):
        """Generate a message uid. Random int is fine per current protocol."""
        return random.randint(1, 127)

    def _publish_server_message(self, dest, uid, ack, msg):
        """Low-level publish - builds and sends one ServerCommunication message."""
        server_msg = ServerCommunication()
        server_msg.src = 1        # Buggy is always src=1
        server_msg.dest = dest    # Server is dest=2
        server_msg.uid = uid
        server_msg.ack = ack
        server_msg.msg = msg
        self.publisher_server.publish(server_msg)

    def send_new_server_message(self, msg_text, purpose):
        """
        Send a brand-new (non-ack) message to the server and remember it
        so we can match the server's ack when it comes back.

        purpose: "PATIENT_FOUND" or "HOSPITAL_FOUND" - tells
        server_communication_callback what to do once this gets acked.
        """
        self.uid = self.generate_uid()
        self.pending_outgoing = {"uid": self.uid, "purpose": purpose}

        self._publish_server_message(dest=2, uid=self.uid, ack=0, msg=msg_text)

        self.get_logger().info(
            f"[SERVER] Sent uid={self.uid} msg='{msg_text}' purpose={purpose} - waiting for ack"
        )
        
    def debug_status(self):
        self.get_logger().info(
            f"Mode={self.lane_mode} | MissionPhase={self.mission_phase} | UID={self.uid} | qrState={self.qr_state}"
        )

        #self.get_logger().info(
        #   f"Pending={self.pending_turn} | "
        #   f"Sign={self.sign_detected} | "
        #   f"TurnNow={self.turn_now}"
        #)

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
        """
        Two-phase turn logic:

        Phase "STRAIGHT":
            Keep driving straight (speed=0.685, turn=0.0) as long as the
            lane is still visible. The moment vector_count drops to 0
            (lane disappears, meaning we've reached the intersection/turn
            point), switch to phase "TURNING".

        Phase "TURNING":
            Apply a fixed turn value (+/- turn_fixed_value depending on
            pending_turn direction) and keep speed at 0.685. Keep doing
            this until the lane reappears (vector_count >= 1), at which
            point the turn is considered complete: reset everything and
            hand control back to normal CENTER lane-following.
        """

        self.target_speed = 0.685

        # ---------------- Phase 1: drive straight until lane disappears ----------------
        if self.turn_phase == "STRAIGHT":
            self.target_turn = 0.0

            if self.edge_vector_count == 0:
                self.turn_phase = "TURNING"
                self.get_logger().info(
                    f"[TURN] Lane lost -> switching to TURNING phase | "
                    f"pending={self.pending_turn}"
                )

        # ---------------- Phase 2: apply fixed turn until lane reappears ----------------
        elif self.turn_phase == "TURNING":

            if self.pending_turn == "Left":
                self.target_turn = self.turn_fixed_value
            elif self.pending_turn == "Right":
                self.target_turn = -self.turn_fixed_value
            else:
                # Safety fallback - no valid pending turn, don't spin blindly
                self.target_turn = 0.0

            self.get_logger().info(
                f"[TURN] Turning | pending={self.pending_turn} | "
                f"edge_count={self.edge_vector_count} | "
                f"target_turn={self.target_turn:.4f}"
            )

            if self.edge_vector_count >= 1:
                # Lane reacquired -> turn is complete
                self.get_logger().info(
                    f"[TURN] Lane reacquired -> turn complete, "
                    f"resuming normal lane following"
                )

                self.turn_now = False
                self.turn_phase = "STRAIGHT"   # reset for next turn
                self.pending_turn = None
                self.target_turn = 0.0
                self.lane_mode = "CENTER"

                # Clear PID history so old turn-time error doesn't leak
                # into fresh lane-following control
                self._reset_pid_state()

    def handle_qr_state(self):
        """
        Vehicle behavior while a QR is being read or just finished.

        READING: QR currently visible -> drive straight (turn=0) so the
                 camera gets a clean, stable read while it's in frame.
        WAITING: QR just left the frame -> stop the vehicle and process
                 the decoded value exactly once, then resume the mission.
        """
        if self.qr_state == "READING":
            #self.target_turn = 0.0
            # Speed left as-is (self.target_speed unchanged).
            # Lower it here if you want a more careful/stable read.
            return

        if self.qr_state == "WAITING":
            # Stop while we decide what to do with the QR result.
            self.target_speed = 0.0
            self.target_turn = 0.0

            self.process_qr_result(self.last_qr_value)

            # Done processing -> hand control back to normal search/driving.
            self.qr_state = "IDLE"
            return

    def process_qr_result(self, qr_value):
        """
        Decide what to do once a QR code has been fully read (called from
        handle_qr_state once qr_is_visible goes True -> False).

        Case (a) - Patient QR (A, B, C, ...), only valid while
                   mission_phase == "SEARCHING_PATIENT":
            Report the patient id to the server and move into
            PATIENT_QR_READ_WAIT_ACK. Do NOT resume driving until the
            full handshake (ack -> hospital location -> our ack) completes.

        Case (b) - Hospital QR (X, Y, Z, ...), only valid while
                   mission_phase == "SEARCHING_HOSPITAL":
            If it matches current_destination -> report to server and
            move into HOSPITAL_QR_READ_WAIT_ACK.
            If it does NOT match -> no state change, keep searching.
        """
        if qr_value is None:
            return

        self.get_logger().info(f"[QR] Processing QR result: {qr_value} (phase={self.mission_phase})")

        # ---------------- Case (a): patient QR ----------------
        if qr_value in self.patient_codes and self.mission_phase == "SEARCHING_PATIENT":
            server_code = self.qr_to_server_patient[qr_value]   # "PATIENT_1" -> "A"
            self.patient_id = server_code

            self.send_new_server_message(server_code, purpose="PATIENT_FOUND")
            self.mission_phase = "PATIENT_QR_READ_WAIT_ACK"
        # ---------------- Case (b): hospital QR ----------------
        elif qr_value in self.hospital_codes and self.mission_phase == "SEARCHING_HOSPITAL":
            server_code = self.qr_to_server_hospital[qr_value]  # "HOSPITAL_1" -> "X"

            if server_code == self.current_destination:   # current_destination stored as "X" from server
                self.send_new_server_message(server_code, purpose="HOSPITAL_FOUND")
                self.mission_phase = "HOSPITAL_QR_READ_WAIT_ACK"
            else:
                self.get_logger().info(f"[QR] '{qr_value}'({server_code}) != target '{self.current_destination}'")
        
        else:
            self.get_logger().warn(
                f"[QR] Unexpected QR '{qr_value}' while in phase {self.mission_phase}"
            )

    def get_hospital_for_patient(self, patient_id):
        """
        STUB: Replace with real logic for mapping a picked-up patient to
        their destination hospital. If the hospital is directly encoded
        in the QR string itself, you likely don't need this lookup at
        all - parse it straight out of qr_value in process_qr_result().
        """
        mapping = {
            "A": "X",
            "B": "Y",
            "C": "Z",
        }
        return mapping.get(patient_id, None)

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