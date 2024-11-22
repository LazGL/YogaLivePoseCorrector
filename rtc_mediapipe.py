import gradio as gr
import cv2
import mediapipe as mp
from gradio_webrtc import WebRTC

# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detection_with_mediapipe(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    return image

css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center'>MediaPipe Pose Detection</h1>")
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream")
        image.stream(
            fn=detection_with_mediapipe, inputs=image, outputs=image, time_limit=10
        )

if __name__ == "__main__":
    demo.launch()