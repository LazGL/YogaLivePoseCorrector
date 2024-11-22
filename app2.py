import gradio as gr
import cv2
from gradio_webrtc import WebRTC
from twilio.rest import Client
import os
from inference import PoseDetector

# Initialize PoseDetector
pose_detector = PoseDetector()

# Twilio RTC configuration
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

if account_sid and auth_token:
    client = Client(account_sid, auth_token)
    token = client.tokens.create()

    rtc_configuration = {
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    }
else:
    rtc_configuration = None

# Global feedback state
feedback_state = {"feedback": ""}


def detect_pose(image):
    """
    Process the image for pose detection and annotate it.
    """
    global feedback_state
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image, feedback = pose_detector.detect_pose(image)
    feedback_state["feedback"] = "\n".join(feedback)  # Update global feedback state
    return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


def update_feedback():
    """
    Fetch the latest feedback from the global state.
    """
    global feedback_state
    return feedback_state["feedback"]


css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""


with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Yoga Pose Correction (Powered by WebRTC ⚡️)
        </h1>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            # Webcam video stream
            video_stream = WebRTC(label="Stream", rtc_configuration=rtc_configuration)

        # Feedback textbox
        feedback_output = gr.Textbox(label="Live Feedback", lines=5)

        # Stream video processing
        video_stream.stream(
            fn=detect_pose,
            inputs=[video_stream],
            outputs=[video_stream],  # Video output only
            time_limit=120,
        )

        # Timer for feedback updates
        feedback_timer = gr.Timer(value=0.5)  # Ticks every 500ms
        feedback_timer.tick(fn=update_feedback, outputs=feedback_output)

if __name__ == "__main__":
    demo.launch()
