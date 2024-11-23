import cv2
import mediapipe as mp
import numpy as np
import time
from calculate_difference import get_all_measurements
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
from sklearn.metrics import accuracy_score

class PoseComparison:
    def __init__(self, reference_image_path, model_name="Qwen/Qwen2.5-0.5B-Instruct", reference_tag="standing", max_new_tokens_value=35):

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
        self.mp_drawing = mp.solutions.drawing_utils

        # Load and process the reference image
        self.reference_landmarks = self.extract_landmarks(cv2.imread(reference_image_path))
        if self.reference_landmarks is None:
            raise FileNotFoundError("The reference image could not be loaded or no pose detected.")

        # Initialize OpenCV for video capture
        self.cap = cv2.VideoCapture(0)

        self.feedback_text = ""
        
        # Initialize timing variables
        self.last_feedback_time = 0
        self.feedback_interval = 2 # Feedback every 3 seconds

        # Variables for FPS calculation
        self.prev_frame_time = 0

        self.higher_accuracy_threshold = 80
        self.lower_accuracy_threshold = 60
        
        self.frame_count = 0
        self.process_every_n_frames = 2  # Adjust 'n' as needed
        
        self.feedback_lock = threading.Lock()
        self.accuracy_score = 0.0
        self.reference_tag = reference_tag
        self.max_new_tokens_value = max_new_tokens_value
        
        self.is_generating_feedback = False

        self.stored_height = 170
        
    ##########
    def overlay_skeletons(self, target_landmarks, user_landmarks, image):
      """
      Overlays the target skeleton on top of the user's skeleton on the given image.

      Args:
          target_landmarks: List of normalized landmarks for the target pose.
          user_landmarks: List of normalized landmarks for the user's pose.
          image: The frame image on which to draw the skeletons.

      Returns:
          Annotated image with both skeletons overlaid.
      """
      image_height, image_width = image.shape[:2]

      # Convert normalized landmarks to pixel coordinates
      target_coords = np.array([
          (lm.x * image_width, lm.y * image_height) for lm in target_landmarks
      ])
      user_coords = np.array([
          (lm.x * image_width, lm.y * image_height) for lm in user_landmarks
      ])

      # Calculate bounding boxes
      target_bbox = cv2.boundingRect(target_coords.astype(np.float32))
      user_bbox = cv2.boundingRect(user_coords.astype(np.float32))

      # Calculate scale factor to match sizes
      scale_factor = min(user_bbox[2] / target_bbox[2], user_bbox[3] / target_bbox[3])

      # Centers of bounding boxes
      target_center = np.array([
          target_bbox[0] + target_bbox[2] / 2,
          target_bbox[1] + target_bbox[3] / 2
      ])
      user_center = np.array([
          user_bbox[0] + user_bbox[2] / 2,
          user_bbox[1] + user_bbox[3] / 2
      ])

      # Scale and translate target landmarks
      aligned_target_coords = (target_coords - target_center) * scale_factor + user_center

      # Draw user's skeleton
      user_image = image.copy()
      self.mp_drawing.draw_landmarks(
          user_image,
          mp.framework.formats.landmark_pb2.LandmarkList(landmark=user_landmarks),
          self.mp_pose.POSE_CONNECTIONS
      )

      # Draw target skeleton on a transparent overlay
      target_image = np.zeros_like(image)
      target_landmark_style = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
      target_connection_style = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)

      # Convert aligned coordinates back to normalized form
      aligned_target_landmarks = []
      for idx, coord in enumerate(aligned_target_coords):
          x_pixel, y_pixel = coord
          normalized_landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark(
              x=float(x_pixel) / image_width,
              y=float(y_pixel) / image_height,
              z=target_landmarks[idx].z  # Use z if depth is relevant
          )
          aligned_target_landmarks.append(normalized_landmark)

      self.mp_drawing.draw_landmarks(
          target_image,
          mp.framework.formats.landmark_pb2.LandmarkList(landmark=aligned_target_landmarks),
          self.mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=target_landmark_style,
          connection_drawing_spec=target_connection_style
      )

      # Overlay the target skeleton onto the user's image
      combined_image = cv2.addWeighted(user_image, 1.0, target_image, 1.0, 0)

      return combined_image
  #########


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
        
        print("FEEDBACK GENERATED CALLED ______________________________________________________________")


        adjustments = self.normalize_and_calculate_adjustments(target_measurements, test_measurements)
        print("_____ __ _____ __ ________", adjustments)
         # Calculate differences and sort by magnitude
        top_adjustments = dict(sorted(adjustments.items(), key=lambda x: abs(x[1]['difference']), reverse=True)[:5])
        
        feedback_lines = [f"{key.replace('_', ' ')}: {values['adjustment']} {values['difference']}" for key, values in top_adjustments.items()]
        print("adjustments", adjustments)
        prompt = "Here are the adjustments to improve your pose:\n" + "\n".join(feedback_lines) + "\nWhat should I focus on? Give ONE single clear ORDER of less than 10 words max. do not say Focus on maintaining balance and alignment."

        messages = [
            {"role": "system", "content": '''You are a yoga instructor. Help the client improve their pose with clear and simple feedback.
If an adjustment is positive (+), suggest actions like 'Lift your arms' or 'Raise your hips.' or 'extend your arms'
If an adjustment is negative (-), suggest actions like 'Lower your hips' or 'Relax your back.' or 'bend your arms
Do not use numbers and focus on a SINGLE clear helpful instruction, the instruction with the HIGHEST PRIORITY per adjustment.'''},
            {"role": "user", "content": prompt}
        ]
        print(prompt)
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

    def get_pose_type_landmarks(self, target_measurements, test_measurements, tag):
        if tag == 'standing':
            relevant_keys = [
                'right_knee_bend', 'left_knee_bend', 'stance_width',
                'right_knee_over_toes', 'left_knee_over_toes',
                'right_elbow_angle', 'left_elbow_angle',
                'shoulder_alignment', 'head_neck_alignment', 'angle_between_legs', 'right_hands_height',
                'left_hand_height', 'hips_in_between_feet', 'stance_width_distance', 'right_foot_distance_from_ground',
                'left_foot_distance_from_ground', 'distance_bestween_feet'
            ]
        elif tag == 'ground':
            relevant_keys = [
                'hip_square', 'back_arch', 'pelvis_tilt',
                'hand_foot_distance', 'core_engagement',
                'spine_vertical', 'symmetry', 'leg_engagement',
                'hip_distance_from_ground', 'hips_in_between_feet', 'right_foot_distance_from_ground',
                'left_foot_distance_from_ground', 'distance_bestween_feet'
            ]
        else:
            relevant_keys = target_measurements.keys()

        return {key: (target_measurements[key], test_measurements[key])
                for key in relevant_keys if key in target_measurements}


    def calculate_accuracy(self, target_measurements, test_measurements, height):
        """
        Calculate the accuracy score based on normalized measurements.
        Ignore the adjustment signs.
        """
        adjustments = self.normalize_and_calculate_adjustments(target_measurements, test_measurements)
        print("________________________________________________")
        #print(adjustments)
        total_measurements = len(adjustments)
        total_error = sum(abs(item['difference']) for item in adjustments.values())  # Use absolute differences

        if total_measurements == 0:
            print("TOTAL MEASUSRLENT = 0")
            return 0.0

        average_error = total_error / total_measurements
        print("REAL RSULT ACCURACY :",  1.0 - average_error)
        accuracy = max(0.0, 1.0 - average_error)  # Accuracy ignores signs
        return accuracy * 100  # Convert to percentage
    # def calculate_accuracy(self, target_measurements, test_measurements):
    #     return 75.00
    
    # def calculate_accuracy(self, target_landmarks, test_landmarks):
        
    #     return accuracy_score(list(target_landmarks.values()), list(test_landmarks.values())) * 100
    
    # def normalize_and_calculate_adjustments(self, target_measurements, test_measurements):
    #     adjustments = {}

    #     # Store height for future use
    #     if target_measurements.get('height') is None:
    #         return adjustments  # Return nothing if height is None
    #     else:
    #         self.stored_height = target_measurements['height']  # Store height for future use

    #     for key in target_measurements:
    #         if key != "height":
    #             if key in test_measurements:
    #                 # Normalize angles (max value = 180)
    #                 if 'angle' in key:
    #                     target_normalized = target_measurements[key] / 180.0 * 100
    #                     test_normalized = test_measurements[key] / 180.0 *100
    #                 # Normalize distances (max value = subject height)
    #                 elif 'distance' in key:
    #                     target_normalized = target_measurements[key] / self.stored_height * 100
    #                     test_normalized = test_measurements[key] / self.stored_height * 100
    #                 else:
    #                     continue  # Skip if not angle or distance

    #                 invert_sign_measurements = [
    #                     'right_knee_over_toes',
    #                     'left_knee_over_toes',
    #                     'right_knee_ankle_alignment',
    #                     'left_knee_ankle_alignment',
    #                     'hip_square',
    #                     'shoulder_alignment',
    #                     'spine_vertical',
    #                     'pelvis_tilt',
    #                     'head_neck_alignment',
    #                     'hip_shoulder_alignment',
    #                     'hips_in_between_feet'
    #                     ]

    #                 # Calculate adjustment sign with inversion where needed
    #                 if key in invert_sign_measurements:
    #                     adjustment_sign = '-' if target_normalized > test_normalized else '+'
    #                 else:
    #                     adjustment_sign = '+' if target_normalized > test_normalized else '-'
                        
    #                 #print(key, adjustment_sign, target_normalized, test_normalized)


    #                 # Store result with normalized values and sign
    #                 adjustments[key] = {
    #                     'target_normalized': int(target_normalized),
    #                     'test_normalized': int(test_normalized),
    #                     'difference': int(abs(target_normalized - test_normalized)),
    #                     'adjustment': adjustment_sign
    #                 }
    #                 print(f"Processed {key}: {adjustments[key]}")
                    
    #                 #print(adjustments)
    #             # Handle the case where key is not in test_measurements if necessary
    #     return adjustments
    def normalize_and_calculate_adjustments(self, target_measurements, test_measurements):
        adjustments = {}

        # Store height for future use
        if target_measurements.get('height') is None:
            return adjustments  # Return empty if height is None
        else:
            self.stored_height = target_measurements['height']  # Store height for future use

        for key in target_measurements:
            if key != "height":
                if key in test_measurements:
                    # Normalize angles (max value = 180)
                    if any(term in key for term in ['angle', 'bend', 'tilt', 'arch']):
                        target_normalized = target_measurements[key] / 180.0 * 100
                        test_normalized = test_measurements[key] / 180.0 * 100
                    # Normalize distances (max value = subject height)
                    elif any(term in key for term in ['distance', 'height', 'width']):
                        target_normalized = target_measurements[key] / self.stored_height * 100
                        test_normalized = test_measurements[key] / self.stored_height * 100
                    else:
                        # For other measurement types, define appropriate normalization
                        # For now, you can skip or handle as needed
                        print(f"Skipping unsupported measurement type: {key}")
                        continue  # Skip unsupported measurement types

                    invert_sign_measurements = [
                        'right_knee_over_toes',
                        'left_knee_over_toes',
                        'right_knee_ankle_alignment',
                        'left_knee_ankle_alignment',
                        'hip_square',
                        'shoulder_alignment',
                        'spine_vertical',
                        'pelvis_tilt',
                        'head_neck_alignment',
                        'hip_shoulder_alignment',
                        'hips_in_between_feet'
                    ]

                    # Calculate adjustment sign with inversion where needed
                    if key in invert_sign_measurements:
                        adjustment_sign = '-' if target_normalized > test_normalized else '+'
                    else:
                        adjustment_sign = '+' if target_normalized > test_normalized else '-'

                    # Store result with normalized values and sign
                    adjustments[key] = {
                        'target_normalized': int(target_normalized),
                        'test_normalized': int(test_normalized),
                        'difference': int(abs(target_normalized - test_normalized)),
                        'adjustment': adjustment_sign
                    }
                    #print(f"Processed {key}: {adjustments[key]}")
                else:
                    print(f"Key {key} not found in test measurements. Skipping.")
            else:
                print("Skipping height key.")

        return adjustments


    def run(self, image):
        try:
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frames != 0:
                # Skip processing to reduce load
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose_video.process(frame_rgb)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                return image, round(self.accuracy_score, 2)

            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_video.process(frame_rgb)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                user_landmarks = self.extract_landmarks(image)
                if self.reference_landmarks is not None and user_landmarks is not None:
                    target_measurements = get_all_measurements(self.reference_landmarks)
                    test_measurements = get_all_measurements(user_landmarks)
                    accuracy_score = self.calculate_accuracy(target_measurements, test_measurements, self.stored_height)
                    print("Accuracy:", accuracy_score)
                    self.accuracy_score = accuracy_score

                    if accuracy_score < self.lower_accuracy_threshold:
                        #self.feedback_text = "Waiting..."
                        None
                        print("1")
                    elif self.lower_accuracy_threshold <= accuracy_score < self.higher_accuracy_threshold:
                        current_time = time.time()
                        print("2")
                        if (current_time - self.last_feedback_time >= self.feedback_interval):
                            #self.last_feedback_time = current_time
                            print("3")
                            self.is_generating_feedback = True  # Set flag to prevent multiple threads
                            #self.feedback_text = "Generating feedback..."
                            relevant_measurements = self.get_pose_type_landmarks(target_measurements, test_measurements, self.reference_tag)
                            threading.Thread(target=self.update_feedback_async, args=(relevant_measurements,)).start()
                            self.last_feedback_time = current_time
                        else:
                            # Do not overwrite self.feedback_text here
                            pass
                    else:  # accuracy >= self.higher_accuracy_threshold
                        self.feedback_text = "Perfect! Hold this position."
                else:
                    #self.feedback_text = "Could not detect landmarks."
                    None
            else:
                self.feedback_text = "No pose detected. Please adjust your position."

            return image, round(self.accuracy_score, 2)
        except Exception as e:
            print("Exception in run:", e)
            return image, round(self.accuracy_score, 2)
    
    # Add this new method to handle asynchronous feedback generation
    def update_feedback_async(self, relevant_measurements):
        try:
            print("Started LLM feedback generation thread")
            llm_output = self.generate_feedback(
                {k: v[0] for k, v in relevant_measurements.items()},
                {k: v[1] for k, v in relevant_measurements.items()},
                self.max_new_tokens_value
            )
            with self.feedback_lock:
                self.feedback_text = llm_output
            print("LLM feedback generated:", llm_output)
        except Exception as e:
            print("Exception in update_feedback_async:", e)
        finally:
            self.is_generating_feedback = False  # Reset the flag
    

