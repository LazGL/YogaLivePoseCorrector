import cv2
import mediapipe as mp
import numpy as np
import time
from calculate_difference import get_all_measurements
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PoseComparison:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", reference_image_path='Tpose_test.png', reference_tag="standing", max_new_tokens_value=35):

        # Load the local LLM model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).eval().to("mps")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_util

        # Load and process the reference image
        self.reference_landmarks = self.extract_landmarks(cv2.imread(reference_image_path))
        if self.reference_landmarks is None:
            raise FileNotFoundError("The reference image could not be loaded or no pose detected.")

        # Initialize OpenCV for video capture
        self.cap = cv2.VideoCapture(0)

        # Initialize timing variables
        self.last_feedback_time = 0
        self.feedback_interval = 3.0  # Feedback every 3 seconds

        # Variables for FPS calculation
        self.prev_frame_time = 0

        self.higher_accuracy_threshold = 70
        self.medium_accuracy_threshold = 80
        self.lower_accuracy_threshold = 90


    def extract_landmarks(self, image):
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
            return landmarks
        return None

    def generate_feedback(self, target_measurements, test_measurements, max_new_tokens_value):
        merged_measurements = {key: {
            'difference': round(abs(target_measurements[key] - test_measurements[key])),
            'adjustment': '+' if target_measurements[key] > test_measurements[key] else '-'
        } for key in target_measurements}

         # Calculate differences and sort by magnitude
        top_adjustments = dict(list(sorted(merged_measurements.items(), key=lambda item: item[1]['difference'], reverse=True))[:5])

        prompt = ("Here are the adjustments to improve your pose:"
                  + '\n'.join([f"{key.replace('_', ' ')}: {values['adjustment']} {values['difference']}"
                               for key, values in top_adjustments.items()])
                  + "\nWhat should I focus on? Give a single clear sentence of less than 10 words max")
        messages = [
            {"role": "system", "content": '''You are a yoga instructor. Help the client improve their pose with clear and simple feedback.
            If an adjustment is positive (+), suggest actions like 'Lift your arms' or 'Raise your hips.'
            If an adjustment is negative (-), suggest actions like 'Lower your hips' or 'Relax your back.'
            Do not use numbers and focus on a single clear instruction, the instruction with the highest priority per adjustment.'''},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens_value,
            do_sample=False,
            use_cache=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def get_pose_type_landmarks(target_measurements, test_measurements, tag):
        if tag == 'standing':
            relevant_keys = [
                'right_knee_bend', 'left_knee_bend', 'stance_width',
                'right_knee_over_toes', 'left_knee_over_toes',
                'right_elbow_angle', 'left_elbow_angle',
                'shoulder_alignment', 'head_neck_alignment'
            ]
        elif tag == 'ground':
            relevant_keys = [
                'hip_square', 'back_arch', 'pelvis_tilt',
                'hand_foot_distance', 'core_engagement',
                'spine_vertical', 'symmetry', 'leg_engagement'
            ]
        else:
            relevant_keys = target_measurements.keys()

        return {key: (target_measurements[key], test_measurements[key])
                for key in relevant_keys if key in target_measurements}


    def calculate_accuracy(self, target_measurements, test_measurements):
        total_measurements = 0
        total_error = 0.0

        for key in target_measurements:
            if key in test_measurements:
                total_measurements += 1
                total_error += abs(target_measurements[key] - test_measurements[key])

        if total_measurements == 0:
            return 0.0

        average_error = total_error / total_measurements
        max_possible_error = 1.0  # Define a normalization scale
        accuracy = max(0.0, 1.0 - (average_error / max_possible_error))

        return accuracy * 100  # Return as a percentage



    def run(self):

        feedback = ["Initializing..."]

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_video.process(frame_rgb)

            current_time = time.time()
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - self.prev_frame_time)

            self.prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                user_landmarks = self.extract_landmarks(frame)
                if self.reference_landmarks is not None and user_landmarks is not None:
                    target_measurements = get_all_measurements(self.reference_landmarks)
                    test_measurements = get_all_measurements(user_landmarks)

                    accuracy_score = self.calculate_accuracy(self, target_measurements, test_measurements)

                    # if not at all in correct position, no call to llm
                    if accuracy_score < self.lower_accuracy_threshold:
                        feedback.append("Waiting...")
                        timer_on = False
                        perfect_start_time = None

                    # if somewhat in correct postion get help from llm on how to position
                    elif self.lower_accuracy_threshold <= accuracy_score < self.higher_accuracy_threshold:
                        if current_time - self.last_feedback_time >= self.feedback_interval:
                            self.last_feedback_time = current_time

                            # get relevant measurments depending on if pose is a standing one or on the ground
                            relevant_measurements = self.get_pose_type_landmarks(target_measurements, test_measurements, self.reference_tag)

                            # Generate feedback messages
                            llm_output = self.generate_feedback(
                                {k: v[0] for k, v in relevant_measurements.items()},  # Target
                                {k: v[1] for k, v in relevant_measurements.items()}   # Test
                            )
                            feedback.append(llm_output)

                        timer_on = False
                        perfect_start_time = None

                    # When in perfect postion need to hold 10s, make boolean timer_on= True
                    else:  # accuracy >= 90
                        if timer_on:
                            if current_time - perfect_start_time >= 10:
                                feedback.append("Perfect! Stay like this for 10 seconds.")
                            else:
                                feedback.append("Hold this position.")
                        else:
                            feedback.append("Hold this position.")
                            timer_on = True
                            perfect_start_time = current_time

                else:
                    feedback.append("Could not detect landmarks in one or both images.")
            else:
                feedback.append("No pose detected. Please adjust your position.")

        #     cv2.putText(frame, self.feedback_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 8)
            cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # cv2.imshow('Pose Comparison', frame)
            # if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        #         break
        # self.cap.release()
        # cv2.destroyAllWindows()

        return cv2, feedback, round(accuracy_score, 2), timer_on
    