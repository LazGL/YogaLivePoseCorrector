"""
Text-to-speech utilities for NamastAI.
Converts feedback and voice cues to audio files for playback.

Strategy:
  1. Try gTTS (Google, online, MP3) — best quality
  2. Fallback: pyttsx3 (offline, WAV) — works without internet
  3. If both fail: return None (app continues silently)
"""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

_temp_file_path = None


def text_to_speech(text):
    """
    Convert text to speech and return the path to the audio file.
    Returns None if all TTS engines fail (allows the app to continue without audio).
    """
    global _temp_file_path

    if not text or not text.strip():
        return None

    # Clean up previous temp file
    if _temp_file_path and os.path.exists(_temp_file_path):
        try:
            os.remove(_temp_file_path)
        except OSError:
            pass
    _temp_file_path = None

    # --- Attempt 1: gTTS (online) ---
    try:
        from gtts import gTTS
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            _temp_file_path = fp.name
        return _temp_file_path
    except Exception as e:
        logger.warning("gTTS failed (no internet?): %s — trying offline fallback", e)

    # --- Attempt 2: pyttsx3 (offline) ---
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)   # Slightly slower for clarity
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            out_path = fp.name
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        _temp_file_path = out_path
        logger.info("pyttsx3 TTS used (offline mode)")
        return _temp_file_path
    except Exception as e:
        logger.error("pyttsx3 TTS also failed: %s", e)

    return None
