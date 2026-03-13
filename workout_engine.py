"""
Workout engine — manages the guided flow state machine.

States: IDLE → CALIBRATING → COUNTDOWN → ACTIVE_POSE → REST → ... → COMPLETE

The engine is ticked every frame by the main app loop. It emits voice cues
and controls which reference pose is loaded into PoseComparison.
"""

import time
import logging
from enum import Enum, auto
from pose_config import POSES, ROUTINES, ACCURACY_PERFECT

logger = logging.getLogger(__name__)


class WorkoutState(Enum):
    IDLE = auto()          # No workout running (manual practice or not started)
    CALIBRATING = auto()   # Waiting for user to be fully visible
    COUNTDOWN = auto()     # Counting down before a pose starts
    ACTIVE_POSE = auto()   # User is doing a pose, receiving feedback
    REST = auto()          # Rest period between poses
    COMPLETE = auto()      # Workout finished


class WorkoutEngine:
    def __init__(self):
        self.state = WorkoutState.IDLE
        self.routine_key = None
        self.pose_queue = []           # List of pose keys remaining
        self.current_pose_key = None
        self.rest_seconds = 10
        self.hold_seconds = 30

        # Timing
        self._state_start_time = 0
        self._hold_start_time = 0      # When user first hit accuracy threshold
        self._holding = False

        # Countdown
        self.countdown_duration = 5
        self._last_countdown_tick = -1

        # Results tracking
        self.results = []              # List of {pose, accuracy, held_for}
        self._pose_accuracies = []     # Accumulate accuracy samples during active pose

        # Voice cue queue — the app reads and clears this
        self.pending_voice_cues = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_routine(self, routine_key):
        """Start a guided workout routine."""
        routine = ROUTINES.get(routine_key)
        if not routine:
            logger.error("Unknown routine: %s", routine_key)
            return

        self.routine_key = routine_key
        self.pose_queue = list(routine["poses"])
        self.rest_seconds = routine.get("rest_seconds", 10)
        self.results = []
        self._pose_accuracies = []

        self._transition(WorkoutState.CALIBRATING)
        self._voice(f"Starting {routine['name']}. Please stand so your full body is visible.")

    def start_manual(self, pose_key):
        """Start manual single-pose practice (no timers, no auto-advance)."""
        self.state = WorkoutState.IDLE
        self.routine_key = None
        self.current_pose_key = pose_key
        self.pose_queue = []
        self.results = []
        return POSES[pose_key]

    def stop(self):
        """Stop any running workout."""
        self.state = WorkoutState.IDLE
        self.routine_key = None
        self.pose_queue = []
        self.current_pose_key = None
        self._voice("Workout stopped.")

    def tick(self, accuracy_score, body_visible):
        """
        Called every processed frame. Returns the current pose config dict
        (or None if idle/complete) so the app knows which reference to use.

        Parameters:
            accuracy_score: float 0-100 from PoseComparison
            body_visible: bool — are landmarks detected?
        """
        now = time.time()
        elapsed = now - self._state_start_time

        if self.state == WorkoutState.IDLE:
            # Manual mode — just return current pose if set
            if self.current_pose_key:
                return POSES.get(self.current_pose_key)
            return None

        elif self.state == WorkoutState.CALIBRATING:
            if body_visible:
                self._advance_to_next_pose()
            elif elapsed > 30:
                self._voice("I still can't see you. Please adjust your phone position.")
                self._state_start_time = now  # Reset timer

        elif self.state == WorkoutState.COUNTDOWN:
            remaining = self.countdown_duration - int(elapsed)
            if remaining != self._last_countdown_tick and remaining > 0:
                self._last_countdown_tick = remaining
                self._voice(str(remaining))
            if elapsed >= self.countdown_duration:
                self._transition(WorkoutState.ACTIVE_POSE)
                self._voice("Go!")
                self._holding = False
                self._hold_start_time = 0
                self._pose_accuracies = []

        elif self.state == WorkoutState.ACTIVE_POSE:
            pose_cfg = POSES.get(self.current_pose_key, {})
            hold_target = pose_cfg.get("hold_seconds", 30)

            self._pose_accuracies.append(accuracy_score)

            # Track how long user holds above threshold
            if accuracy_score >= ACCURACY_PERFECT:
                if not self._holding:
                    self._holding = True
                    self._hold_start_time = now
                held_for = now - self._hold_start_time
                if held_for >= hold_target:
                    avg_acc = sum(self._pose_accuracies) / len(self._pose_accuracies)
                    self.results.append({
                        "pose": self.current_pose_key,
                        "accuracy": round(avg_acc, 1),
                        "held_for": round(held_for, 1),
                    })
                    self._voice("Great job! Relax.")
                    if self.pose_queue:
                        self._transition(WorkoutState.REST)
                    else:
                        self._transition(WorkoutState.COMPLETE)
                        self._announce_summary()
            else:
                self._holding = False

            # Timeout: if pose takes too long (3x hold target), move on
            if elapsed > hold_target * 3:
                avg_acc = sum(self._pose_accuracies) / len(self._pose_accuracies) if self._pose_accuracies else 0
                self.results.append({
                    "pose": self.current_pose_key,
                    "accuracy": round(avg_acc, 1),
                    "held_for": 0,
                })
                self._voice("Let's move on.")
                if self.pose_queue:
                    self._transition(WorkoutState.REST)
                else:
                    self._transition(WorkoutState.COMPLETE)
                    self._announce_summary()

        elif self.state == WorkoutState.REST:
            remaining = self.rest_seconds - int(elapsed)
            if remaining == 3 and self._last_countdown_tick != 3:
                self._last_countdown_tick = 3
                self._voice("Get ready.")
            if elapsed >= self.rest_seconds:
                self._advance_to_next_pose()

        elif self.state == WorkoutState.COMPLETE:
            pass

        if self.current_pose_key:
            return POSES.get(self.current_pose_key)
        return None

    def get_status_text(self):
        """Return a short status string for the UI overlay."""
        now = time.time()
        elapsed = now - self._state_start_time

        if self.state == WorkoutState.IDLE:
            if self.current_pose_key:
                return f"Free Practice: {POSES[self.current_pose_key]['name']}"
            return "Ready"

        elif self.state == WorkoutState.CALIBRATING:
            return "Stand so your full body is visible..."

        elif self.state == WorkoutState.COUNTDOWN:
            remaining = max(0, self.countdown_duration - int(elapsed))
            pose_name = POSES.get(self.current_pose_key, {}).get("name", "")
            return f"Next: {pose_name} in {remaining}..."

        elif self.state == WorkoutState.ACTIVE_POSE:
            pose_cfg = POSES.get(self.current_pose_key, {})
            pose_name = pose_cfg.get("name", "")
            hold_target = pose_cfg.get("hold_seconds", 30)
            if self._holding:
                held = int(now - self._hold_start_time)
                return f"{pose_name} — Hold {held}/{hold_target}s"
            return f"{pose_name} — Adjust your pose"

        elif self.state == WorkoutState.REST:
            remaining = max(0, self.rest_seconds - int(elapsed))
            return f"Rest — {remaining}s"

        elif self.state == WorkoutState.COMPLETE:
            return "Workout Complete!"

        return ""

    def pop_voice_cues(self):
        """Return and clear pending voice cues."""
        cues = self.pending_voice_cues[:]
        self.pending_voice_cues.clear()
        return cues

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transition(self, new_state):
        logger.info("Workout state: %s -> %s", self.state.name, new_state.name)
        self.state = new_state
        self._state_start_time = time.time()
        self._last_countdown_tick = -1

    def _advance_to_next_pose(self):
        if not self.pose_queue:
            self._transition(WorkoutState.COMPLETE)
            self._announce_summary()
            return
        self.current_pose_key = self.pose_queue.pop(0)
        pose_name = POSES[self.current_pose_key]["name"]
        self._voice(f"Next pose: {pose_name}")
        self._transition(WorkoutState.COUNTDOWN)

    def _voice(self, text):
        self.pending_voice_cues.append(text)
        logger.info("Voice cue: %s", text)

    def _announce_summary(self):
        if not self.results:
            self._voice("Workout complete.")
            return
        total = len(self.results)
        avg = sum(r["accuracy"] for r in self.results) / total
        self._voice(f"Workout complete! You did {total} poses with an average accuracy of {int(avg)} percent.")
