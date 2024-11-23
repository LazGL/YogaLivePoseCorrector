
# import gradio as gr
# import cv2
# from gradio_webrtc import WebRTC
# from inference_demo import PoseDetector
# from tts_utils import text_to_speech  # Import the TTS function
# from css_style import css

# # Initialize PoseDetector
# pose_detector = PoseDetector()

# # Hardcoded index for the highlighted pictogram
# current_pictogram_index = 1

# # List of pictograms (ensure these paths are correct and accessible)
# pictograms = [
#     "images_front_end/input-onlinepngtools.png",
#     "images_front_end/input-onlinepngtools-3.png",
#     "images_front_end/input-onlinepngtools-2.png",
#     "images_front_end/input-onlinepngtools-4.png",
# ]

# # Global feedback state
# feedback_state = {"feedback": ""}
# last_feedback_text = None  # Global variable to store the last feedback text


# def detect_pose(image):
#     """
#     Process the image for pose detection and annotate it.
#     """
#     global feedback_state
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     annotated_image, feedback = pose_detector.detect_pose(image)
#     feedback_state["feedback"] = "\n".join(feedback)  # Update global feedback state
#     return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


# def update_feedback():
#     """
#     Fetch the latest feedback from the global state and generate audio if the feedback has changed.
#     """
#     global feedback_state, last_feedback_text
#     feedback_text = feedback_state['feedback']
#     feedback_html = f"<div style='text-align: center;'>{feedback_text}</div>"

#     # Check if the feedback text has changed
#     if feedback_text != last_feedback_text:
#         # Generate audio from the feedback text
#         audio_data = text_to_speech(feedback_text)
#         # Update last_feedback_text
#         last_feedback_text = feedback_text
#     else:
#         # No change in feedback; do not update audio_data
#         audio_data = None  # Returning None prevents the audio from replaying

#     return feedback_html, audio_data  # Return both HTML and audio

# def get_current_pose():
#     """
#     Retrieve the current pose from the PoseDetector.
#     """
#     return pose_detector.current_pose


# with gr.Blocks(css=css) as demo:
#     gr.HTML(
#         """
#         <h1 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
#         🧘 Yoga Pose Correction 🧘
#         </h1>
#         """
#     )
#     with gr.Row():  # Create two columns
#         with gr.Column(scale=1, elem_classes=["left-column"]):  # Left column for webcam and feedback
#             gr.HTML(
#                 """
#                 <h2 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
#                 Welcome to NamastAI
#                 </h2>
#                 """
#             )
#             with gr.Group(elem_classes=["my-group"]):
#                 # Webcam video stream
#                 video_stream = WebRTC(label="Stream", elem_id="custom-stream")

#             # Feedback displayed below the stream
#             with gr.Column(elem_classes=["my-column"]):
#                 feedback_output = gr.HTML(elem_id="custom-textbox")
#                 # Audio output for TTS
#                 audio_output = gr.Audio(type="filepath", label="Feedback Audio", autoplay=True, elem_id="feedback-audio")

#             # Stream video processing
#             video_stream.stream(
#                 fn=detect_pose,
#                 inputs=[video_stream],
#                 outputs=[video_stream],
#                 time_limit=120,
#             )

#         with gr.Column(scale=1, elem_id="pictogram-container"):  # Right column for pictograms
#             gr.HTML(
#                 """
#                 <h2 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
#                 Today's Program
#                 </h2>
#                 """
#             )
#             current_pose_output = gr.Text(label="Pose", elem_id="current-pose-display")

#             with gr.Column(elem_id="pictogram-container"):
#                 for i, pictogram in enumerate(pictograms):
#                     is_highlighted = i+1 == current_pictogram_index
#                     gr.Image(
#                         value=pictogram,
#                         type="filepath",
#                         elem_classes=["highlighted" if is_highlighted else "pictogram"],
#                         label='Pose'
#                     )

#     # Timer for feedback updates
#     feedback_timer = gr.Timer(value=5)  # Ticks every 3 seconds
#     feedback_timer.tick(fn=update_feedback, outputs=[feedback_output, audio_output])

#     feedback_timer.tick(
#         fn=get_current_pose,  # Update pose dynamically
#         outputs=[current_pose_output],
#     )

# if __name__ == "__main__":
#     demo.launch()

######


import gradio as gr
import cv2
from gradio_webrtc import WebRTC
from inference_demo import PoseDetector
from tts_utils import text_to_speech  # Import the TTS function
from css_style import css

# Initialize PoseDetector
pose_detector = PoseDetector()

# List of pictograms (ensure these paths are correct and accessible)
pictograms = [
    "images_front_end/input-onlinepngtools.png",  # T Pose
    "images_front_end/input-onlinepngtools-3.png",  # Warrior Pose
    "images_front_end/input-onlinepngtools-2.png",
    "images_front_end/input-onlinepngtools-4.png",
]

# Global feedback state
feedback_state = {"feedback": ""}
last_feedback_text = None  # Global variable to store the last feedback text


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
    Fetch the latest feedback from the global state and generate audio if the feedback has changed.
    """
    global feedback_state, last_feedback_text
    feedback_text = feedback_state['feedback']
    feedback_html = f"<div style='text-align: center;'>{feedback_text}</div>"

    # Check if the feedback text has changed
    if feedback_text != last_feedback_text:
        # Generate audio from the feedback text
        audio_data = text_to_speech(feedback_text)
        # Update last_feedback_text
        last_feedback_text = feedback_text
    else:
        # No change in feedback; do not update audio_data
        audio_data = None  # Returning None prevents the audio from replaying

    return feedback_html, audio_data  # Return both HTML and audio


def get_current_pose():
    """
    Retrieve the current pose from the PoseDetector.
    """
    return pose_detector.current_pose


def get_pictogram_outputs():
    """
    Dynamically reconstruct pictograms with the correct highlighted class.
    """
    current_pose = pose_detector.current_pose
    current_pictogram_index = 0
    if current_pose == "T Pose":
        current_pictogram_index = 1  # Highlight T Pose
    elif current_pose == "Warrior Pose":
        current_pictogram_index = 2  # Highlight Warrior Pose

    # Rebuild the pictograms with updated classes
    updated_pictograms = [
        gr.Image(
            value=pictogram,
            type="filepath",
            elem_classes=["highlighted" if (i + 1) == current_pictogram_index else "pictogram"],
            interactive=False,
        )
        for i, pictogram in enumerate(pictograms)
    ]
    return updated_pictograms


with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
        🧘 Yoga Pose Correction 🧘
        </h1>
        """
    )
    with gr.Row():  # Create two columns
        with gr.Column(scale=1, elem_classes=["left-column"]):  # Left column for webcam and feedback
            gr.HTML(
                """
                <h2 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
                Welcome to NamastAI
                </h2>
                """
            )
            with gr.Group(elem_classes=["my-group"]):
                # Webcam video stream
                video_stream = WebRTC(label="Stream", elem_id="custom-stream")

            # Feedback displayed below the stream
            with gr.Column(elem_classes=["my-column"]):
                feedback_output = gr.HTML(elem_id="custom-textbox")
                # Audio output for TTS
                audio_output = gr.Audio(type="filepath", label="Feedback Audio", autoplay=True, elem_id="feedback-audio")

            # Stream video processing
            video_stream.stream(
                fn=detect_pose,
                inputs=[video_stream],
                outputs=[video_stream],
                time_limit=120,
            )

        with gr.Column(scale=1, elem_id="pictogram-container"):  # Right column for pictograms
            gr.HTML(
                """
                <h2 style='text-align: center; font-family: "Poppins", sans-serif; color: #2C3E50;'>
                Today's Program
                </h2>
                """
            )
            current_pose_output = gr.Text(label="Pose", elem_id="current-pose-display")

            # Pictogram outputs
            pictogram_outputs = [
                gr.Image(value=pictogram, type="filepath", elem_classes=["pictogram"], interactive=False)
                for pictogram in pictograms
            ]

    # Timer for feedback updates
    feedback_timer = gr.Timer(value=5)  # Ticks every 5 seconds
    feedback_timer.tick(fn=update_feedback, outputs=[feedback_output, audio_output])

    # Update the current pose display
    feedback_timer.tick(
        fn=get_current_pose,
        outputs=[current_pose_output],
    )

    # Dynamically reconstruct pictograms
    feedback_timer.tick(
        fn=get_pictogram_outputs,
        outputs=pictogram_outputs,
    )

if __name__ == "__main__":
    demo.launch(share=True)
