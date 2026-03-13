"""
Text-to-speech utilities for NamastAI.
Converts feedback and voice cues to audio files for playback.
"""

import logging
import os
import tempfile
from gtts import gTTS

logger = logging.getLogger(__name__)

_temp_file_path = None


def text_to_speech(text):
    """
    Convert text to speech and return the path to the audio file.
    Returns None if TTS fails (allows the app to continue without audio).
    """
    global _temp_file_path

    if not text or not text.strip():
        return None

    try:
        tts = gTTS(text)
        # Clean up previous temp file
        if _temp_file_path and os.path.exists(_temp_file_path):
            try:
                os.remove(_temp_file_path)
            except OSError:
                pass
        # Save to a new temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            _temp_file_path = fp.name
        return _temp_file_path
    except Exception as e:
        logger.error("TTS failed: %s", e)
        return None
