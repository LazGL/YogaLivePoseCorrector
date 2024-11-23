import gradio as gr
import cv2
from gradio_webrtc import WebRTC
from inference import PoseDetector

# Initialize PoseDetector
pose_detector = PoseDetector()

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
    # Return HTML with centered text
    return f"<div style='text-align: center;'>{feedback_state['feedback']}</div>"


css = """
.my-group {
    max-width: 600px !important;
    max-height: 600px !important;
}
.my-column {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
#custom-textbox div {
    font-size: 32px;
    color: #708090; /* Text color */
    background-color: #ffffff; /* White background */
    border: none !important;
    border-radius: 8px; /* Rounded corners */
    padding: 12px; /* Internal padding */
    width: 700px !important; /* Force reduced width */
    max-width: 700px !important;
    height: auto; /* Auto-adjust height */
    overflow-y: auto; /* Add vertical scroll if needed */
    margin: 0 auto; /* Center the div horizontally */
}
.gradio-container {
    background-color: #ffffff !important; /* Set background to white */
    border: none !important; /* Remove faint border around components */
}
#custom-stream .webrtc-container {
    border: none !important; /* Remove border around WebRTC stream */
    box-shadow: none !important; /* Remove shadow */
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Yoga Pose Correction
        </h1>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            # Webcam video stream
            video_stream = WebRTC(label="Stream", elem_id="custom-stream")

        # Feedback displayed below the stream
        with gr.Column(elem_classes=["my-column"]):
            feedback_output = gr.HTML(elem_id="custom-textbox")

        # Stream video processing
        video_stream.stream(
            fn=detect_pose,
            inputs=[video_stream],
            outputs=[video_stream],  # Video output only
            time_limit=120,
        )

    # Timer for feedback updates
    feedback_timer = gr.Timer(value=3)  # Ticks every 3 seconds
    feedback_timer.tick(fn=update_feedback, outputs=feedback_output)

if __name__ == "__main__":
    demo.launch()
