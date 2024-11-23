import cv2
import mediapipe as mp
import numpy as np
import time
from calculate_difference import get_all_measurements
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

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
        self.feedback_interval = 3 # Feedback every 3 seconds

        # Variables for FPS calculation
        self.prev_frame_time = 0

        self.higher_accuracy_threshold = 90
        self.medium_accuracy_threshold = 80
        self.lower_accuracy_threshold = 50
        
        self.frame_count = 0
        self.process_every_n_frames = 5  # Adjust 'n' as needed
        
        self.feedback_lock = threading.Lock()
        self.accuracy_score = 0.0
        self.reference_tag = reference_tag
        self.max_new_tokens_value = max_new_tokens_value
        
        self.is_generating_feedback = False


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
        merged_measurements = {key: {
            'difference': round(abs(target_measurements[key] - test_measurements[key])),
            'adjustment': '+' if target_measurements[key] > test_measurements[key] else '-'
        } for key in target_measurements}

        adjustments = self.normalize_and_calculate_adjustments(target_measurements, test_measurements)
        
         # Calculate differences and sort by magnitude
        top_adjustments = dict(sorted(adjustments.items(), key=lambda x: abs(x[1]['difference']), reverse=True)[:5])
        
        feedback_lines = [f"{key.replace('_', ' ')}: {values['adjustment']} {values['difference']}" for key, values in top_adjustments.items()]

        prompt = "Here are the adjustments to improve your pose:\n" + "\n".join(feedback_lines) + "\nWhat should I focus on? Give a single clear sentence of less than 10 words max"

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

    def get_pose_type_landmarks(self, target_measurements, test_measurements, tag):
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


    # def calculate_accuracy(self, target_measurements, test_measurements, height):
    #     """
    #     Calculate the accuracy score based on normalized measurements.
    #     Ignore the adjustment signs.
    #     """
    #     adjustments = self.normalize_and_calculate_adjustments(target_measurements, test_measurements, height)
    #     total_measurements = len(adjustments)
    #     total_error = sum(item['difference'] for item in adjustments.values())  # Use absolute differences

    #     if total_measurements == 0:
    #         return 0.0

    #     average_error = total_error / total_measurements
    #     accuracy = max(0.0, 1.0 - average_error)  # Accuracy ignores signs
    #     return accuracy * 100  # Convert to percentage
    def calculate_accuracy(self, target_measurements, test_measurements):
        return 75.00
    
    def normalize_and_calculate_adjustments(self, target_measurements, test_measurements):
        adjustments = {}

        for key in target_measurements:
            if key != "height":
                if key in test_measurements:
                    # Normalize angles (max value = 180)
                    if 'angle' in key:
                        target_normalized = target_measurements[key] / 180.0
                        test_normalized = test_measurements[key] / 180.0
                    # Normalize distances (max value = subject height)
                    elif 'distance' in key:
                        target_normalized = target_measurements[key] / target_measurements['height']
                        test_normalized = test_measurements[key] / test_measurements['height']
                    else:
                        continue  # Skip if not angle or distance

                    # Calculate adjustment sign
                    adjustment_sign = '+' if target_normalized > test_normalized else '-'

                    # Store result with normalized values and sign
                    adjustments[key] = {
                        'target_normalized': round(target_normalized, 3),
                        'test_normalized': round(test_normalized, 3),
                        'difference': round(abs(target_normalized - test_normalized), 3),
                        'adjustment': adjustment_sign
                    }
                # Handle the case where key is not in test_measurements if necessary
        return adjustments

    # def run(self, image):
    #     try: 
            
            
    #         accuracy_score = 0.0 
    #         print("1") #yesss
            
            
    #         self.frame_count += 1
    #         if self.frame_count % self.process_every_n_frames != 0:
    #             # Skip processing to reduce load
                
    #             ########## To Comment Out if too computaionally intesive 
    #                 # Convert the input image to RGB
    #             frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #             results = self.pose_video.process(frame_rgb)
    #             print("2") # yesss

    #             if results.pose_landmarks:
    #                 # Draw landmarks on the image
    #                 self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
    #             ###############
    #             return image, round(self.accuracy_score,2)


    #         # Convert the input image to RGB
    #         frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         results = self.pose_video.process(frame_rgb)
    #         print("3") ## yess
            
    #         if results.pose_landmarks:
    #             print("Pose landmarks detected")################# debug
    #             print("4")### yes 
    #             # Draw landmarks on the image
    #             self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    #             # Extract landmarks from the user's pose
    #             user_landmarks = self.extract_landmarks(image)
    #             #if self.reference_landmarks is not None and user_landmarks is not None:
    #             if True : 
    #                 # Calculate measurements for the reference and user poses
    #                 print("5") # yessss
    #                 target_measurements = get_all_measurements(self.reference_landmarks)
    #                 test_measurements = get_all_measurements(user_landmarks)

    #                 # Calculate the accuracy score
    #                 accuracy_score = self.calculate_accuracy(target_measurements, test_measurements)
    #                 self.accuracy_score = accuracy_score
    #                 print("Calculated accuracy score:", accuracy_score) ########################debug 
    #                 print("6")

    #                 # Determine feedback based on accuracy score
    #                 if accuracy_score < self.lower_accuracy_threshold:
    #                     self.feedback_text = "Waiting..."
    #                     print("7")
    #                 elif self.lower_accuracy_threshold <= accuracy_score < self.higher_accuracy_threshold:
    #                     #self.feedback_text = "Good enough"
    #                     print("8")
    #                     # Generate feedback using the LLM
    #                     current_time = time.time()
    #                     if current_time - self.last_feedback_time >= self.feedback_interval:
    #                     #if True:
    #                         print("9")
    #                         print("Starting LLM feedback generation")
    #                         self.last_feedback_time = current_time  # Update last feedback time
    #                         relevant_measurements = self.get_pose_type_landmarks(target_measurements, test_measurements, self.reference_tag)
                            
    #                         # Create a new thread to run the LLM
    #                         threading.Thread(target=self.update_feedback_async, args=(relevant_measurements,)).start()
                        
    #                 else: # accuracy >= 90
    #                     print("10")
    #                     self.feedback_text = "Perfect! Hold this position."
    #                     print("over 90 accuracy score")
    #             else:
    #                 print("11")
    #                 #self.feedback_text = "Could not detect landmarks in one or both images."
    #         else:
    #             print("12")
    #             self.feedback_text = "No pose detected. Please adjust your position."

    #         self.accuracy_score = accuracy_score######################### what is the use ? trying to debug 
    #         print("Returning from run method")
    #         print("13")
    #         # Return the processed image, feedback, accuracy score, and timer status
    #         return image, round(self.accuracy_score, 2)
    #     except Exception as e:
    #         print("Exception in run:", e)
    #         print("14")
    #         return image, round(self.accuracy_score, 2)

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
                    accuracy_score = self.calculate_accuracy(target_measurements, test_measurements)
                    self.accuracy_score = accuracy_score

                    if accuracy_score < self.lower_accuracy_threshold:
                        #self.feedback_text = "Waiting..."
                        None
                    elif self.lower_accuracy_threshold <= accuracy_score < self.higher_accuracy_threshold:
                        current_time = time.time()
                        if (current_time - self.last_feedback_time >= self.feedback_interval) and not self.is_generating_feedback:
                            #self.last_feedback_time = current_time
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
    