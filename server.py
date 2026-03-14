"""
NamastAI — FastAPI server (replaces Gradio app_llm.py)

Architecture:
  /ws/video    — Browser sends JPEG frames → server annotates → returns JPEG
  /ws/feedback — Server pushes JSON at 10 Hz (accuracy, status, voice, audio)
  POST /session/start  — Start guided routine or free practice
  POST /session/stop   — Stop session
  GET  /session/history — Return recent sessions as JSON
  GET  /config         — Return pose/routine config for the frontend
  Static files from static/ and images_front_end/
"""

import asyncio
import base64
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_new import PoseComparison
from workout_engine import WorkoutEngine, WorkoutState
from tts_utils import text_to_speech
from session_tracker import save_session, load_history
from pose_config import POSES, ROUTINES, ACCURACY_PERFECT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NamastAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend and pose images
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images_front_end", StaticFiles(directory="images_front_end"), name="images")

# Thread pool for blocking frame processing (max 1 worker — no parallel frame processing)
_executor = ThreadPoolExecutor(max_workers=1)

# Pre-cache valid poses (pictogram file must exist)
_VALID_POSES = {k: v for k, v in POSES.items() if os.path.exists(v.get("pictogram", ""))}

_STAGNANT_THRESHOLD = 2  # Re-voice feedback after N cycles without accuracy improvement


# ---------------------------------------------------------------------------
# Application state (centralised to avoid scattered globals)
# ---------------------------------------------------------------------------

class _AppState:
    def __init__(self):
        self.pose_detector: Optional[PoseComparison] = None
        self.workout: WorkoutEngine = WorkoutEngine()

        # Thread safety
        self._detector_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._processing_lock = threading.Lock()
        self._processing: bool = False

        # TTS background state
        self.pending_audio: Optional[bytes] = None
        self.pending_audio_type: str = "mp3"
        self._tts_thread: Optional[threading.Thread] = None

        # Correction-repeat tracking
        self.last_voice_text: Optional[str] = None
        self._last_feedback_accuracy: Optional[float] = None
        self._stagnant_cycles: int = 0

        # Session persistence
        self._last_workout_state: Optional[WorkoutState] = None


state = _AppState()


# ---------------------------------------------------------------------------
# Detector management
# ---------------------------------------------------------------------------

def _ensure_detector(reference_image_path: str, reference_tag: str = "standing") -> bool:
    with state._detector_lock:
        if (
            state.pose_detector is None
            or getattr(state.pose_detector, "_ref_path", None) != reference_image_path
        ):
            if not os.path.exists(reference_image_path):
                logger.warning("Reference image not found: %s", reference_image_path)
                return False
            try:
                logger.info("Loading pose detector for %s ...", reference_image_path)
                state.pose_detector = PoseComparison(
                    reference_image_path=reference_image_path,
                    reference_tag=reference_tag,
                )
                state.pose_detector._ref_path = reference_image_path
                logger.info("Detector ready (tag=%s)", reference_tag)
            except Exception as e:
                logger.error("Failed to load pose detector: %s", e)
                return False
    return True


# ---------------------------------------------------------------------------
# Frame processing (runs in thread pool — blocking MediaPipe call)
# ---------------------------------------------------------------------------

def _process_frame_sync(img_bytes: bytes) -> Optional[bytes]:
    """
    Decode JPEG → run pose detection → return annotated JPEG bytes.
    Returns None if a frame is already being processed (drop policy).
    """
    with state._processing_lock:
        if state._processing:
            return None  # Drop frame — still busy with previous
        state._processing = True

    try:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None or state.pose_detector is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_rgb, accuracy = state.pose_detector.run(img_rgb)

        # Tick workout engine
        body_visible = accuracy > 0
        if state.workout.state != WorkoutState.IDLE:
            pose_cfg = state.workout.tick(accuracy, body_visible)
            if pose_cfg and pose_cfg.get("reference_image") != getattr(
                state.pose_detector, "_ref_path", None
            ):
                _ensure_detector(pose_cfg["reference_image"], pose_cfg.get("tag", "standing"))

        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes()

    except Exception as e:
        logger.error("Frame processing error: %s", e)
        return None
    finally:
        with state._processing_lock:
            state._processing = False


# ---------------------------------------------------------------------------
# TTS (background thread — non-blocking)
# ---------------------------------------------------------------------------

def _tts_worker(text: str) -> None:
    path = text_to_speech(text)
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                audio_bytes = f.read()
            audio_type = "wav" if path.endswith(".wav") else "mp3"
            with state._audio_lock:
                state.pending_audio = audio_bytes
                state.pending_audio_type = audio_type
        except OSError as e:
            logger.error("Could not read TTS file: %s", e)


def _trigger_tts(text: str) -> None:
    """Start TTS in background only if previous TTS has finished."""
    if state._tts_thread is None or not state._tts_thread.is_alive():
        state._tts_thread = threading.Thread(
            target=_tts_worker, args=(text,), daemon=True
        )
        state._tts_thread.start()


# ---------------------------------------------------------------------------
# Feedback payload builder (called at 10 Hz by /ws/feedback)
# ---------------------------------------------------------------------------

def _build_feedback_payload() -> dict:
    if state.pose_detector is None:
        return {
            "accuracy": 0,
            "status": "Select a mode to begin",
            "feedback": "",
            "active_pose": None,
            "completed_poses": [],
            "audio_b64": None,
            "audio_type": None,
        }

    with state.pose_detector.feedback_lock:
        feedback_text = state.pose_detector.feedback_text
    with state.pose_detector._accuracy_lock:
        accuracy = state.pose_detector.accuracy_score

    voice_cues = state.workout.pop_voice_cues()
    status_text = state.workout.get_status_text()

    # Auto-save session on COMPLETE transition
    if (
        state.workout.state == WorkoutState.COMPLETE
        and state._last_workout_state != WorkoutState.COMPLETE
        and state.workout.results
    ):
        routine_name = (
            ROUTINES.get(state.workout.routine_key, {}).get("name")
            if state.workout.routine_key
            else None
        )
        save_session(routine_name, state.workout.results)
    state._last_workout_state = state.workout.state

    # Correction-repeat: re-voice feedback if accuracy stagnates
    if 0 < accuracy < ACCURACY_PERFECT:
        if (
            state._last_feedback_accuracy is not None
            and accuracy <= state._last_feedback_accuracy + 2
        ):
            state._stagnant_cycles += 1
        else:
            state._stagnant_cycles = 0
        state._last_feedback_accuracy = accuracy
        if state._stagnant_cycles >= _STAGNANT_THRESHOLD:
            state.last_voice_text = None  # Allow re-voicing
            state._stagnant_cycles = 0
    else:
        state._stagnant_cycles = 0
        state._last_feedback_accuracy = accuracy

    # Decide what to speak
    text_to_speak = None
    if voice_cues:
        text_to_speak = ". ".join(voice_cues)
    elif feedback_text and feedback_text != state.last_voice_text and feedback_text.strip():
        text_to_speak = feedback_text

    if text_to_speak:
        state.last_voice_text = text_to_speak
        _trigger_tts(text_to_speak)

    # Grab ready audio (clears it so it's only sent once)
    audio_b64 = None
    audio_type = None
    with state._audio_lock:
        if state.pending_audio:
            audio_b64 = base64.b64encode(state.pending_audio).decode()
            audio_type = state.pending_audio_type
            state.pending_audio = None

    completed_poses = [r["pose"] for r in state.workout.results] if state.workout.results else []

    return {
        "accuracy": round(accuracy, 1),
        "status": status_text,
        "feedback": feedback_text,
        "active_pose": state.workout.current_pose_key,
        "completed_poses": completed_poses,
        "audio_b64": audio_b64,
        "audio_type": audio_type,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/video")
async def video_websocket(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    logger.info("Video WebSocket connected")
    try:
        while True:
            data = await ws.receive_bytes()
            annotated = await loop.run_in_executor(_executor, _process_frame_sync, data)
            if annotated:
                await ws.send_bytes(annotated)
    except WebSocketDisconnect:
        logger.info("Video WebSocket disconnected")
    except Exception as e:
        logger.error("Video WebSocket error: %s", e)


@app.websocket("/ws/feedback")
async def feedback_websocket(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("Feedback WebSocket connected")
    try:
        while True:
            payload = _build_feedback_payload()
            await ws.send_json(payload)
            await asyncio.sleep(0.1)  # Push at 10 Hz (vs Gradio's 0.33 Hz)
    except WebSocketDisconnect:
        logger.info("Feedback WebSocket disconnected")
    except Exception as e:
        logger.error("Feedback WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    mode: str   # "guided" or "free"
    key: str    # routine key or pose key


@app.post("/session/start")
def session_start(req: SessionStartRequest) -> dict:
    if req.mode == "guided":
        routine = ROUTINES.get(req.key)
        if not routine:
            return JSONResponse({"error": "Unknown routine"}, status_code=400)
        first_pose = POSES[routine["poses"][0]]
        _ensure_detector(first_pose["reference_image"], first_pose.get("tag", "standing"))
        state.workout.start_routine(req.key)
        return {"status": f"Starting {routine['name']}..."}

    if req.mode == "free":
        pose_cfg = POSES.get(req.key)
        if not pose_cfg:
            return JSONResponse({"error": "Unknown pose"}, status_code=400)
        _ensure_detector(pose_cfg["reference_image"], pose_cfg.get("tag", "standing"))
        state.workout.start_manual(req.key)
        return {"status": f"Free Practice: {pose_cfg['name']}"}

    return JSONResponse({"error": "Unknown mode"}, status_code=400)


@app.post("/session/stop")
def session_stop() -> dict:
    state.workout.stop()
    return {"status": "Stopped"}


@app.get("/session/history")
def session_history() -> list:
    return load_history(max_sessions=5)


@app.get("/config")
def get_config() -> dict:
    """Return pose and routine config for the frontend JS."""
    poses_js = {}
    for key, pose in POSES.items():
        picto = pose.get("pictogram", "")
        poses_js[key] = {
            "name": pose["name"],
            "pictogram_url": f"/{picto}" if os.path.exists(picto) else None,
        }
    routines_js = {
        key: {"name": r["name"], "description": r.get("description", "")}
        for key, r in ROUTINES.items()
    }
    return {"poses": poses_js, "routines": routines_js}


@app.get("/")
def root() -> HTMLResponse:
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
