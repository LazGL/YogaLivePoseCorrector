import threading
import queue
import time
import gradio as gr
import cv2
from gradio_webrtc import WebRTC
from inference import PoseDetector
from tts_test import Dimits  # Importation de Dimits

# Initialisation du PoseDetector
pose_detector = PoseDetector()
rtc_configuration = None

# Initialisation de la queue pour stocker les feedbacks
feedback_queue = queue.Queue()

# Initialisation du moteur TTS via Dimits
parent_destn = "/chemin/vers/le/dossier"  # Remplacez par le chemin où vous souhaitez stocker le cache
dt = Dimits(parent_destn)  # Initialisation de Dimits avec le chemin de cache

# Fonction de détection de poses
def detect_pose(image):
    """
    Process the image for pose detection and annotate it.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image, feedback = pose_detector.detect_pose(image)

    # Ajout du feedback dans la queue pour traitement
    if feedback:
        feedback_queue.put("\n".join(feedback))

    return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# Fonction pour lire et jouer le feedback depuis la queue
def tts_worker():
    """
    Thread worker to fetch feedback from the queue and perform TTS.
    """
    while True:
        try:
            feedback = feedback_queue.get(timeout=10)  # Attendre 10s pour un nouvel élément
            if feedback:
                print(f"Playing TTS: {feedback}")
                dt.text_2_speech(feedback, engine="say")  # Utiliser ta fonction de TTS
            feedback_queue.task_done()
        except queue.Empty:
            pass  # Rien à lire, continue d'attendre

# Thread séparé pour TTS
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Fonction pour récupérer le dernier feedback
def update_feedback():
    """
    Fetch the latest feedback from the queue (for Gradio UI update).
    """
    try:
        feedback = feedback_queue.get_nowait()
        feedback_queue.task_done()
        return feedback
    except queue.Empty:
        return "No feedback yet."

# Interface utilisateur avec Gradio
css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Yoga Pose Correction (TTS Integrated ⚡️)
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
            outputs=[video_stream],
            time_limit=120,
        )

        # Timer pour mettre à jour le feedback
        feedback_timer = gr.Timer(value=10)  # Mise à jour toutes les 10 secondes
        feedback_timer.tick(fn=update_feedback, outputs=feedback_output)

if __name__ == "__main__":
    demo.launch()