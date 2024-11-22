# tts_utils.py

from gtts import gTTS
import tempfile
import os

temp_file_path = None  # Global variable to store the temp file path

def text_to_speech(text):
    """
    Convert text to speech and return the path to the audio file.
    """
    global temp_file_path
    tts = gTTS(text)
    # Remove the previous temp file if it exists
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    # Save to a new temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        temp_file_path = fp.name
    return temp_file_path
