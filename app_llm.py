"""
NamastAI — AI Yoga Coach
Main application entry point with two modes:
  1. Guided Flow: auto-advancing workout with voice cues, countdowns, rest periods
  2. Free Practice: pick a single pose and get continuous feedback
"""

import logging
import os
import gradio as gr
import cv2
from gradio_webrtc import WebRTC
from inference_new import PoseComparison
from tts_utils import text_to_speech
from css_style import css
from pose_config import POSES, ROUTINES, ACCURACY_PERFECT, ACCURACY_FEEDBACK
from workout_engine import WorkoutEngine, WorkoutState
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pose_detector = None          # Initialized when user starts a session
workout = WorkoutEngine()
last_voice_text = None        # Track last voiced feedback to avoid replaying

# We lazily initialize pose_detector because we need a reference image first
_detector_lock = threading.Lock()


def _ensure_detector(reference_image_path, reference_tag="standing"):
    """Create or re-create the PoseComparison with a new reference image."""
    global pose_detector
    with _detector_lock:
        if pose_detector is None or getattr(pose_detector, '_ref_path', None) != reference_image_path:
            if not os.path.exists(reference_image_path):
                logger.warning("Reference image not found: %s", reference_image_path)
                return False
            try:
                pose_detector = PoseComparison(
                    reference_image_path=reference_image_path,
                    reference_tag=reference_tag,
                )
                pose_detector._ref_path = reference_image_path
                logger.info("Loaded reference: %s (tag=%s)", reference_image_path, reference_tag)
            except Exception as e:
                logger.error("Failed to load pose detector: %s", e)
                return False
    return True


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def detect_pose(image):
    """Process each webcam frame: detect pose, run workout engine, annotate."""
    global pose_detector

    if pose_detector is None:
        return image

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image, accuracy_score = pose_detector.run(image)

        body_visible = accuracy_score > 0

        # Tick the workout engine (handles guided flow logic)
        if workout.state != WorkoutState.IDLE:
            pose_cfg = workout.tick(accuracy_score, body_visible)
            # If workout switched poses, reload reference
            if pose_cfg and pose_cfg.get("reference_image") != getattr(pose_detector, '_ref_path', None):
                _ensure_detector(pose_cfg["reference_image"], pose_cfg.get("tag", "standing"))

        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error("Exception in detect_pose: %s", e)
        return image


# ---------------------------------------------------------------------------
# Feedback updates (ticked by timer)
# ---------------------------------------------------------------------------

def update_feedback():
    """Called every few seconds by gr.Timer — returns HTML feedback, audio, accuracy, status, pictograms."""
    global last_voice_text

    if pose_detector is None:
        return _feedback_html("Select a mode to begin"), None, _accuracy_html(0), "Ready", _pictograms_html(None, [])

    # Get pose feedback (thread-safe reads)
    with pose_detector.feedback_lock:
        feedback_text = pose_detector.feedback_text
    with pose_detector._accuracy_lock:
        accuracy = pose_detector.accuracy_score

    # Get workout engine voice cues (these take priority)
    voice_cues = workout.pop_voice_cues()
    status_text = workout.get_status_text()

    # Determine what to speak
    audio_data = None
    if voice_cues:
        combined = ". ".join(voice_cues)
        audio_data = text_to_speech(combined)
        last_voice_text = combined
    elif feedback_text and feedback_text != last_voice_text and feedback_text.strip():
        audio_data = text_to_speech(feedback_text)
        last_voice_text = feedback_text

    # Build pictogram state
    completed_poses = [r["pose"] for r in workout.results] if workout.results else []
    active_pose = workout.current_pose_key
    picto_html = _pictograms_html(active_pose, completed_poses)

    return _feedback_html(feedback_text), audio_data, _accuracy_html(accuracy), status_text, picto_html


def _feedback_html(text):
    if not text:
        text = ""
    return f"<div style='text-align: center;'>{text}</div>"


def _pictograms_html(active_pose_key, completed_pose_keys):
    """Generate pictogram grid HTML with highlight/completed state."""
    html = "<div style='display: flex; flex-direction: column; align-items: center; gap: 10px;'>"
    for key, pose in POSES.items():
        pictogram_path = pose.get("pictogram", "")
        if not os.path.exists(pictogram_path):
            continue

        if key == active_pose_key:
            border = "3px solid #27ae60"
            opacity = "1.0"
            shadow = "0 4px 12px rgba(39, 174, 96, 0.3)"
            max_w = "140px"
            label_color = "#27ae60"
        elif key in completed_pose_keys:
            border = "3px solid #95a5a6"
            opacity = "0.7"
            shadow = "none"
            max_w = "120px"
            label_color = "#95a5a6"
        else:
            border = "2px solid transparent"
            opacity = "0.5"
            shadow = "none"
            max_w = "120px"
            label_color = "#7f8c8d"

        html += f"""
        <div style='text-align: center;'>
            <img src='/file={pictogram_path}'
                 style='border-radius: 12px; border: {border}; opacity: {opacity};
                        max-width: {max_w}; box-shadow: {shadow};
                        transition: all 0.3s ease-in-out;' />
            <div style='font-size: 13px; color: {label_color}; font-weight: 600; margin-top: 4px;'>
                {pose["name"]}
            </div>
        </div>"""
    html += "</div>"
    return html


def _accuracy_html(score):
    """Large, color-coded accuracy number visible from distance."""
    score = round(score, 0)
    if score >= ACCURACY_PERFECT:
        color_class = "accuracy-high"
    elif score >= ACCURACY_FEEDBACK:
        color_class = "accuracy-mid"
    else:
        color_class = "accuracy-low"

    return f"""
    <div id='accuracy-display'>
        <div class='accuracy-number {color_class}'>{int(score)}%</div>
        <div class='accuracy-label'>Accuracy</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def start_guided(routine_key):
    """Start a guided workout routine."""
    routine = ROUTINES.get(routine_key)
    if not routine:
        return "Unknown routine"

    first_pose_key = routine["poses"][0]
    first_pose = POSES[first_pose_key]
    _ensure_detector(first_pose["reference_image"], first_pose.get("tag", "standing"))

    workout.start_routine(routine_key)
    return f"Starting {routine['name']}..."


def start_free_practice(pose_key):
    """Start free practice on a single pose."""
    pose_cfg = POSES.get(pose_key)
    if not pose_cfg:
        return "Unknown pose"

    _ensure_detector(pose_cfg["reference_image"], pose_cfg.get("tag", "standing"))
    workout.start_manual(pose_key)
    return f"Free Practice: {pose_cfg['name']}"


def stop_session():
    """Stop any running workout."""
    workout.stop()
    return "Stopped"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(css=css, title="NamastAI") as demo:
        gr.HTML("""
            <h1 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
            NamastAI
            </h1>
        """)

        with gr.Row():
            # ===== LEFT COLUMN: Video + Feedback =====
            with gr.Column(scale=2, elem_classes=["left-column"]):
                with gr.Group(elem_classes=["my-group"]):
                    video_stream = WebRTC(label="Stream", elem_id="custom-stream")

                # Status line (workout state, countdown, hold timer)
                status_output = gr.HTML(elem_id="status-display", value="Select a mode to begin")

                # Large accuracy display
                accuracy_output = gr.HTML(value=_accuracy_html(0))

                # Feedback text
                feedback_output = gr.HTML(elem_id="custom-textbox")

                # Audio (hidden but autoplay)
                audio_output = gr.Audio(
                    type="filepath", label="Feedback Audio",
                    autoplay=True, elem_id="feedback-audio", visible=False,
                )

                video_stream.stream(
                    fn=detect_pose,
                    inputs=[video_stream],
                    outputs=[video_stream],
                    time_limit=600,
                )

            # ===== RIGHT COLUMN: Mode Selection =====
            with gr.Column(scale=1):
                gr.HTML("""
                    <h2 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
                    Choose Your Mode
                    </h2>
                """)

                # Guided Routines
                with gr.Group(elem_id="mode-panel"):
                    gr.HTML("<h3 style='margin:0 0 8px 0;'>Guided Workouts</h3>")
                    for key, routine in ROUTINES.items():
                        btn = gr.Button(
                            f"{routine['name']}",
                            variant="primary",
                            size="lg",
                        )
                        btn.click(fn=lambda k=key: start_guided(k), outputs=[status_output])

                # Free Practice
                with gr.Group(elem_id="mode-panel"):
                    gr.HTML("<h3 style='margin:0 0 8px 0;'>Free Practice</h3>")
                    for key, pose in POSES.items():
                        btn = gr.Button(
                            pose["name"],
                            variant="secondary",
                            size="lg",
                        )
                        btn.click(fn=lambda k=key: start_free_practice(k), outputs=[status_output])

                # Stop button
                stop_btn = gr.Button("Stop Session", variant="stop", size="lg")
                stop_btn.click(fn=stop_session, outputs=[status_output])

                # Pose pictograms (dynamically updated via timer)
                gr.HTML("""
                    <h3 style='text-align: center; margin-top: 20px; color: #2C3E50;'>
                    Pose Reference
                    </h3>
                """)
                pictogram_output = gr.HTML(
                    value=_pictograms_html(None, []),
                    elem_id="pictogram-container",
                )

        # Timer for feedback/status updates
        feedback_timer = gr.Timer(value=3)
        feedback_timer.tick(
            fn=update_feedback,
            outputs=[feedback_output, audio_output, accuracy_output, status_output, pictogram_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
