import mediapipe as mp
import math
import cv2
from utils import calculate_angle

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )

        # Updated reference angles for the desired pose
        self.reference_pose = {
            "left_elbow_angle": 180,
            "right_elbow_angle": 180,
            "left_arm_elevation": 90,
            "right_arm_elevation": 90,
            "left_leg_angle": 180,
            "right_leg_angle": 180,
        }

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
            threshold = 20  # degrees

            # Compare elbow angles
            left_elbow_deviation = abs(left_elbow_angle - self.reference_pose["left_elbow_angle"])
            if left_elbow_deviation > threshold:
                feedback.append("Straighten your left arm.")
                deviations.append(left_elbow_deviation)

            right_elbow_deviation = abs(right_elbow_angle - self.reference_pose["right_elbow_angle"])
            if right_elbow_deviation > threshold:
                feedback.append("Straighten your right arm.")
                deviations.append(right_elbow_deviation)

            # Compare arm elevation angles
            left_arm_elevation_deviation = abs(left_arm_elevation - self.reference_pose["left_arm_elevation"])
            if left_arm_elevation_deviation > threshold:
                feedback.append("Raise your left arm to shoulder level.")
                deviations.append(left_arm_elevation_deviation)

            right_arm_elevation_deviation = abs(right_arm_elevation - self.reference_pose["right_arm_elevation"])
            if right_arm_elevation_deviation > threshold:
                feedback.append("Raise your right arm to shoulder level.")
                deviations.append(right_arm_elevation_deviation)

            # Compare leg angles
            left_leg_deviation = abs(left_leg_angle - self.reference_pose["left_leg_angle"])
            if left_leg_deviation > threshold:
                feedback.append("Keep your left leg straight.")
                deviations.append(left_leg_deviation)

            right_leg_deviation = abs(right_leg_angle - self.reference_pose["right_leg_angle"])
            if right_leg_deviation > threshold:
                feedback.append("Keep your right leg straight.")
                deviations.append(right_leg_deviation)

        # Determine the feedback message
        if deviations:
            # Select the feedback with the largest deviation
            max_index = deviations.index(max(deviations))
            feedback = [feedback[max_index]]
        else:
            # Provide positive feedback if the pose is correct
            feedback = ["Perfect, hold this pose"]

        return image, feedback
