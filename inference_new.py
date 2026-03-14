import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from calculate_difference import get_all_measurements
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

# MediaPipe: try multiple import paths across versions
# v0.9 and earlier:  mp.solutions.pose  (attribute on the package)
# v0.10.x:           mediapipe.python.solutions.pose
# v0.10.14+/0.11+:   mediapipe.solutions.pose  (flat, no python/ subpackage)
try:
    from mediapipe.python.solutions import pose as _mp_pose_module
    from mediapipe.python.solutions import drawing_utils as _mp_drawing_module
except (ImportError, ModuleNotFoundError):
    try:
        from mediapipe.solutions import pose as _mp_pose_module          # type: ignore
        from mediapipe.solutions import drawing_utils as _mp_drawing_module  # type: ignore
    except (ImportError, ModuleNotFoundError):
        import mediapipe as _mp_pkg
        _mp_pose_module = _mp_pkg.solutions.pose          # type: ignore
        _mp_drawing_module = _mp_pkg.solutions.drawing_utils  # type: ignore

try:
    from mediapipe.framework.formats import landmark_pb2
except (ImportError, ModuleNotFoundError):
    from google.protobuf import descriptor as _  # ensure protobuf available
    import mediapipe.framework.formats.landmark_pb2 as landmark_pb2  # type: ignore

logger = logging.getLogger(__name__)

# Pre-computed key sets for O(1) lookup in normalize_and_calculate_adjustments()
# (replaces repeated `any(term in key for term in [...])` string scanning per measurement)
_ANGLE_KEYS = frozenset([
    'right_knee_bend', 'left_knee_bend',
    'right_knee_over_toes', 'left_knee_over_toes',
    'right_knee_ankle_alignment', 'left_knee_ankle_alignment',
    'back_arch', 'pelvis_tilt',
    'right_elbow_angle', 'left_elbow_angle',
    'angle_between_legs',
    'hip_square', 'shoulder_alignment', 'spine_vertical',
    'head_neck_alignment', 'hip_shoulder_alignment',
])
_DISTANCE_KEYS = frozenset([
    'stance_width', 'stance_width_distance',
    'distance_between_feet',
    'right_hands_height', 'left_hand_height', 'hip_height',
    'right_foot_distance_from_ground', 'left_foot_distance_from_ground',
    'hips_in_between_feet',
])
_INVERT_SIGN_KEYS = frozenset([
    'right_knee_over_toes', 'left_knee_over_toes',
    'right_knee_ankle_alignment', 'left_knee_ankle_alignment',
    'hip_square', 'shoulder_alignment', 'spine_vertical',
    'pelvis_tilt', 'head_neck_alignment', 'hip_shoulder_alignment',
    'hips_in_between_feet',
])


class PoseComparison:
    def __init__(self, reference_image_path, model_name="Qwen/Qwen2.5-0.5B-Instruct",
                 reference_tag="standing", max_new_tokens_value=35, device=None,
                 preloaded_model=None, preloaded_tokenizer=None):

        # Detect compute device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # Reuse pre-loaded model if provided, otherwise load from scratch
        if preloaded_model is not None and preloaded_tokenizer is not None:
            self.model = preloaded_model
            self.tokenizer = preloaded_tokenizer
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            ).eval().to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize MediaPipe Pose
        self.mp_pose = _mp_pose_module
        self.pose = _mp_pose_module.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.pose_video = _mp_pose_module.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = _mp_drawing_module

        # Load and process the reference image
        self.reference_landmarks = self.extract_landmarks(cv2.imread(reference_image_path))
        if self.reference_landmarks is None:
            raise FileNotFoundError(
                f"Reference image could not be loaded or no pose detected: {reference_image_path}"
            )

        self.feedback_text = ""

    def switch_reference(self, reference_image_path: str, reference_tag: str = "standing"):
        """Switch to a different pose reference without reloading the LLM."""
        new_landmarks = self.extract_landmarks(cv2.imread(reference_image_path))
        if new_landmarks is None:
            raise FileNotFoundError(
                f"Reference image could not be loaded or no pose detected: {reference_image_path}"
            )
        self.reference_landmarks = new_landmarks
        self.reference_tag = reference_tag
        # Reset state for new pose
        with self.feedback_lock:
            self.feedback_text = ""
        with self._accuracy_lock:
            self.accuracy_score = 0.0
        self.last_feedback_time = 0
        self._feedback_thread = None
        logger.info("Switched reference to %s (tag=%s)", reference_image_path, reference_tag)

        # Timing
        self.last_feedback_time = 0
        self.feedback_interval = 2  # seconds between feedback updates

        # Accuracy thresholds
        self.higher_accuracy_threshold = 80
        self.lower_accuracy_threshold = 60

        # Frame processing
        self.frame_count = 0
        self.process_every_n_frames = 2

        # Thread safety
        self.feedback_lock = threading.Lock()
        self.is_generating_feedback = False

        self.accuracy_score = 0.0
        self._accuracy_lock = threading.Lock()  # Guards accuracy_score reads/writes
        self.reference_tag = reference_tag
        self.max_new_tokens_value = max_new_tokens_value
        self.stored_height = 170

        # B3: Keep reference to LLM feedback thread to prevent concurrent spawning
        self._feedback_thread: threading.Thread | None = None
        
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
          landmark_pb2.LandmarkList(landmark=user_landmarks),
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
          normalized_landmark = landmark_pb2.NormalizedLandmark(
              x=float(x_pixel) / image_width,
              y=float(y_pixel) / image_height,
              z=target_landmarks[idx].z  # Use z if depth is relevant
          )
          aligned_target_landmarks.append(normalized_landmark)

      self.mp_drawing.draw_landmarks(
          target_image,
          landmark_pb2.LandmarkList(landmark=aligned_target_landmarks),
          self.mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=target_landmark_style,
          connection_drawing_spec=target_connection_style
      )

      # Overlay the target skeleton onto the user's image
      combined_image = cv2.addWeighted(user_image, 1.0, target_image, 1.0, 0)

      return combined_image

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
        adjustments = self.normalize_and_calculate_adjustments(target_measurements, test_measurements)
        # Calculate differences and sort by magnitude
        top_adjustments = dict(sorted(adjustments.items(), key=lambda x: abs(x[1]['difference']), reverse=True)[:5])

        feedback_lines = [f"{key.replace('_', ' ')}: {values['adjustment']} {values['difference']}" for key, values in top_adjustments.items()]
        prompt = "Here are the adjustments to improve your pose:\n" + "\n".join(feedback_lines) + "\nWhat should I focus on? Give ONE single clear ORDER of less than 10 words max. do not say Focus on maintaining balance and alignment."

        messages = [
            {"role": "system", "content": '''You are a yoga instructor. Help the client improve their pose with clear and simple feedback.
If an adjustment is positive (+), suggest actions like 'Lift your arms' or 'Raise your hips.' or 'extend your arms'
If an adjustment is negative (-), suggest actions like 'Lower your hips' or 'Relax your back.' or 'bend your arms
Do not use numbers and focus on a SINGLE clear helpful instruction, the instruction with the HIGHEST PRIORITY per adjustment.'''},
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
                'shoulder_alignment', 'head_neck_alignment', 'angle_between_legs', 'right_hands_height',
                'left_hand_height', 'hips_in_between_feet', 'stance_width_distance', 'right_foot_distance_from_ground',
                'left_foot_distance_from_ground', 'distance_between_feet'
            ]
        elif tag == 'ground':
            relevant_keys = [
                'hip_square', 'back_arch', 'pelvis_tilt',
                'hand_foot_distance', 'core_engagement',
                'spine_vertical', 'symmetry', 'leg_engagement',
                'hip_distance_from_ground', 'hips_in_between_feet', 'right_foot_distance_from_ground',
                'left_foot_distance_from_ground', 'distance_between_feet'
            ]
        else:
            relevant_keys = target_measurements.keys()

        return {key: (target_measurements[key], test_measurements[key])
                for key in relevant_keys if key in target_measurements}


    def calculate_accuracy(self, target_measurements, test_measurements, height):
        """
        Calculate the accuracy score based on normalized measurements.
        Returns a percentage 0-100 where 100 is a perfect match.
        """
        adjustments = self.normalize_and_calculate_adjustments(target_measurements, test_measurements)
        total_measurements = len(adjustments)
        total_error = sum(abs(item['difference']) for item in adjustments.values())

        if total_measurements == 0:
            return 0.0

        average_error = total_error / total_measurements
        accuracy = max(0.0, 1.0 - average_error)
        return accuracy * 100

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
                    if key in _ANGLE_KEYS:
                        target_normalized = target_measurements[key] / 180.0 * 100
                        test_normalized = test_measurements[key] / 180.0 * 100
                    # Normalize distances (max value = subject height)
                    elif key in _DISTANCE_KEYS:
                        target_normalized = target_measurements[key] / self.stored_height * 100
                        test_normalized = test_measurements[key] / self.stored_height * 100
                    else:
                        continue  # Skip unsupported measurement types

                    # Calculate adjustment sign with inversion where needed (O(1) lookup)
                    if key in _INVERT_SIGN_KEYS:
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
                else:
                    logger.debug("Key %s not found in test measurements", key)

        return adjustments


    def run(self, image):
        """
        Process a single frame (already RGB from app_llm).
        Returns (annotated_image, accuracy_score).
        """
        try:
            self.frame_count += 1

            # image is already RGB (converted in app_llm.py) — feed directly to MediaPipe
            results = self.pose_video.process(image)

            if self.frame_count % self.process_every_n_frames != 0:
                # Lightweight frame: just draw skeleton, reuse last accuracy
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                with self._accuracy_lock:
                    score = self.accuracy_score
                return image, round(score, 2)

            # Full analysis frame
            if results.pose_landmarks:
                user_landmarks_raw = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ])

                if self.reference_landmarks is not None and user_landmarks_raw is not None:
                    target_measurements = get_all_measurements(self.reference_landmarks)
                    test_measurements = get_all_measurements(user_landmarks_raw)
                    accuracy_score = self.calculate_accuracy(
                        target_measurements, test_measurements, self.stored_height
                    )
                    with self._accuracy_lock:
                        self.accuracy_score = accuracy_score

                    # B1: Only overlay the REFERENCE skeleton when accuracy needs work.
                    # (Removed redundant first call that passed identical user→user landmarks.)
                    if accuracy_score < self.higher_accuracy_threshold:
                        try:
                            ref_landmarks = [
                                landmark_pb2.NormalizedLandmark(
                                    x=float(pt[0]), y=float(pt[1]), z=float(pt[2])
                                )
                                for pt in self.reference_landmarks
                            ]
                            image = self.overlay_skeletons(
                                ref_landmarks,
                                list(results.pose_landmarks.landmark),
                                image,
                            )
                        except Exception as e:
                            logger.debug("Overlay fallback: %s", e)
                            self.mp_drawing.draw_landmarks(
                                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                            )
                    else:
                        self.mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                        )

                    # Generate LLM feedback when accuracy is mid-range
                    if accuracy_score < self.lower_accuracy_threshold:
                        pass  # Too far off — no useful feedback
                    elif accuracy_score < self.higher_accuracy_threshold:
                        current_time = time.time()
                        if current_time - self.last_feedback_time >= self.feedback_interval:
                            # B3: Only spawn a new thread if the previous one has finished
                            if self._feedback_thread is None or not self._feedback_thread.is_alive():
                                relevant = self.get_pose_type_landmarks(
                                    target_measurements, test_measurements, self.reference_tag
                                )
                                self._feedback_thread = threading.Thread(
                                    target=self.update_feedback_async, args=(relevant,), daemon=True
                                )
                                self._feedback_thread.start()
                                self.last_feedback_time = current_time
                    else:
                        with self.feedback_lock:
                            self.feedback_text = "Perfect! Hold this position."
                else:
                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                    )
            else:
                with self.feedback_lock:
                    self.feedback_text = "No pose detected. Please adjust your position."

            with self._accuracy_lock:
                score = self.accuracy_score
            return image, round(score, 2)
        except Exception as e:
            logger.error("Exception in run: %s", e)
            with self._accuracy_lock:
                score = self.accuracy_score
            return image, round(score, 2)

    def _rule_based_feedback(self, relevant_measurements):
        """
        Fallback feedback when the LLM is unavailable.
        Picks the measurement with the largest difference and returns a simple instruction.
        """
        best_key = None
        best_diff = 0
        best_sign = "+"
        for key, values in relevant_measurements.items():
            diff = abs(values[0] - values[1]) if len(values) >= 2 else 0
            if diff > best_diff:
                best_diff = diff
                best_key = key
                best_sign = "+" if values[0] > values[1] else "-"

        if best_key is None:
            return "Keep adjusting your pose."

        readable = best_key.replace("_", " ")
        action = "Increase" if best_sign == "+" else "Decrease"
        return f"{action} your {readable}."

    def update_feedback_async(self, relevant_measurements):
        try:
            llm_output = self.generate_feedback(
                {k: v[0] for k, v in relevant_measurements.items()},
                {k: v[1] for k, v in relevant_measurements.items()},
                self.max_new_tokens_value
            )
            with self.feedback_lock:
                self.feedback_text = llm_output
            logger.debug("LLM feedback: %s", llm_output)
        except Exception as e:
            logger.warning("LLM feedback failed, using rule-based fallback: %s", e)
            fallback = self._rule_based_feedback(relevant_measurements)
            with self.feedback_lock:
                self.feedback_text = fallback
        finally:
            self.is_generating_feedback = False

