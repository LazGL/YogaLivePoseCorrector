import mediapipe as mp
import math
import cv2
from utils import calculate_angle

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing_utils = mp.solutions.drawing_utils
        self.mp_drawing_style = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

        self.reference_pose = {
            "left_arm_angle": 90,  # Ideal angle between left shoulder, elbow, and wrist
            "right_arm_angle": 180,  # Ideal angle for the right arm
            "left_leg_angle": 90,  # Angle between hip, knee, and ankle
        }

    def detect_pose(self, image):
        # Process the image with MediaPipe Pose
        result = self.pose.process(image)

        feedback = []
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

            # Calculate angles for feedback
            left_arm_angle = calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST],
            )
            right_arm_angle = calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST],
            )
            left_leg_angle = calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE],
            )

            # Compare with reference pose
            if abs(left_arm_angle - self.reference_pose["left_arm_angle"]) > 10:
                feedback.append("Raise your left arm to 90 degrees.")
            if abs(right_arm_angle - self.reference_pose["right_arm_angle"]) > 10:
                feedback.append("Straighten your right arm.")
            if abs(left_leg_angle - self.reference_pose["left_leg_angle"]) > 10:
                feedback.append("Bend your left knee more.")

        return image, feedback
