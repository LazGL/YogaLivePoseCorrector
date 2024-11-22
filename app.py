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


def detect_pose_with_feedback(image):
    """
    Process the image to detect poses and embed feedback into the video frame.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image, feedback = pose_detector.detect_pose(image)

    # Add feedback text to the video frame
    for i, line in enumerate(feedback):
        cv2.putText(
            annotated_image,
            line,
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


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
            video_stream = WebRTC(label="Stream", rtc_configuration=rtc_configuration)

        # Stream video processing with feedback embedded
        video_stream.stream(
            fn=detect_pose_with_feedback,
            inputs=[video_stream],
            outputs=[video_stream],  # Single WebRTC output
            time_limit=10,
        )

if __name__ == "__main__":
    demo.launch()

