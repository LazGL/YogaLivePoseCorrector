import mediapipe as mp
import math
import time  # For tracking time
from utils import calculate_angle


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing_utils.DrawingSpec(
            color=(173, 216, 230), thickness=2, circle_radius=2
        )

        # Reference angles for T Pose
        self.reference_t_pose = {
            "left_elbow_angle": 180,
            "right_elbow_angle": 180,
            "left_arm_elevation": 90,
            "right_arm_elevation": 90,
        }

        # Reference angles for Warrior Pose
        self.reference_warrior_pose = {
            "left_elbow_angle": 180,
            "right_elbow_angle": 180,
            "left_arm_elevation": 0,  # Arms directly above head
            "right_arm_elevation": 0,
            "left_leg_angle": 90,  # Front leg bent at 90 degrees
            "right_leg_angle": 180,  # Back leg straight
        }

        self.current_pose = "T Pose"  # Start with the T Pose
        self.pose_start_time = None  # Time when the pose is correctly achieved
        self.pose_hold_duration = 3  # Required hold duration for each pose (seconds)

    def calculate_arm_elevation_angle(self, shoulder, wrist):
        """
        Calculate the angle between the arm vector (shoulder to wrist) and the vertical axis.
        Returns angle in degrees:
        - 0 degrees: arm pointing up
        - 90 degrees: arm horizontal
        - 180 degrees: arm pointing down
        """
        dx = wrist.x - shoulder.x
        dy = shoulder.y - wrist.y  # Adjusted for image coordinate system
        angle_rad = math.atan2(abs(dx), dy)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def detect_pose(self, image):
        # Process the image with MediaPipe Pose
        result = self.pose.process(image)

        feedback = []
        deviations = []

        if result.pose_landmarks:
            # Draw landmarks
            self.mp_drawing_utils.draw_landmarks(
                image,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.drawing_spec,
            )

            # Extract keypoints
            landmarks = result.pose_landmarks.landmark

            # Select the appropriate reference pose
            if self.current_pose == "T Pose":
                reference_pose = self.reference_t_pose
            elif self.current_pose == "Warrior Pose":
                reference_pose = self.reference_warrior_pose

            # Calculate elbow angles
            left_elbow_angle = calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST],
            )
            right_elbow_angle = calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST],
            )

            # Calculate arm elevation angles
            left_arm_elevation = self.calculate_arm_elevation_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST],
            )
            right_arm_elevation = self.calculate_arm_elevation_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST],
            )

            # Calculate leg angles
            left_leg_angle = calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE],
            )
            right_leg_angle = calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE],
            )

            # Threshold for acceptable deviation
            threshold = 35  # degrees


            # Compare angles and deviations
            if self.current_pose == "T Pose":
                # Check deviations for T Pose
                left_elbow_deviation = abs(left_elbow_angle - reference_pose["left_elbow_angle"])
                if left_elbow_deviation > threshold:
                    feedback.append("Straighten your left arm.")
                    deviations.append(left_elbow_deviation)

                right_elbow_deviation = abs(right_elbow_angle - reference_pose["right_elbow_angle"])
                if right_elbow_deviation > threshold:
                    feedback.append("Straighten your right arm.")
                    deviations.append(right_elbow_deviation)

                left_arm_elevation_deviation = abs(left_arm_elevation - reference_pose["left_arm_elevation"])
                if left_arm_elevation_deviation > threshold:
                    feedback.append("Raise your left arm to shoulder level.")
                    deviations.append(left_arm_elevation_deviation)

                right_arm_elevation_deviation = abs(right_arm_elevation - reference_pose["right_arm_elevation"])
                if right_arm_elevation_deviation > threshold:
                    feedback.append("Raise your right arm to shoulder level.")
                    deviations.append(right_arm_elevation_deviation)

            elif self.current_pose == "Warrior Pose":
      # Check bent leg and straight leg conditions with orientation
                bent_leg_threshold = 45  # Degrees tolerance for a bent leg
                straight_leg_threshold = 30  # Degrees tolerance for a straight leg

                # # Get coordinates for left and right knees
                # left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                # right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]

                # Ensure the left leg is the bent leg
                if abs(left_leg_angle - 90) > bent_leg_threshold:
                    feedback.append("Bend your left leg closer to 90 degrees.")
                    deviations.append(abs(left_leg_angle - 90))
                if abs(right_leg_angle - 180) > straight_leg_threshold:
                    feedback.append("Straighten your right leg fully.")
                    deviations.append(abs(right_leg_angle - 180))
      

                # Arm conditions: Check if arms are straight and vertically above head
                if abs(left_elbow_angle - 180) > straight_leg_threshold:
                    feedback.append("Straighten your left arm.")
                    deviations.append(abs(left_elbow_angle - 180))

                if abs(right_elbow_angle - 180) > straight_leg_threshold:
                    feedback.append("Straighten your right arm.")
                    deviations.append(abs(right_elbow_angle - 180))

                if abs(left_arm_elevation - 0) > straight_leg_threshold:
                    feedback.append("Raise your left arm vertically above your head.")
                    deviations.append(abs(left_arm_elevation - 0))

                if abs(right_arm_elevation - 0) > straight_leg_threshold:
                    feedback.append("Raise your right arm vertically above your head.")
                    deviations.append(abs(right_arm_elevation - 0))



            # Select the largest deviation feedback
            if deviations:
                max_index = deviations.index(max(deviations))
                feedback = [feedback[max_index]]
                self.pose_start_time = None  # Reset the timer if deviations occur
            else:
                # Timer logic for correct pose
                if self.pose_start_time is None:
                    self.pose_start_time = time.time()  # Start the timer
                elapsed_time = time.time() - self.pose_start_time

                if elapsed_time >= self.pose_hold_duration:
                    if self.current_pose == "T Pose":
                        feedback = ["T Pose complete! Now transition to Warrior Pose."]
                        self.current_pose = "Warrior Pose"  # Transition to Warrior Pose
                        self.pose_start_time = None  # Reset timer for next pose
                    elif self.current_pose == "Warrior Pose":
                        feedback = ["Warrior Pose complete! Great job!"]
                        self.pose_start_time = None  # Final pose, no transition
                else:
                    feedback = [
                        f"Hold the {self.current_pose} for {self.pose_hold_duration - int(elapsed_time)} more seconds."
                    ]

        return image, feedback
