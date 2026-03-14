"""
Session tracking for NamastAI.

Saves completed workout sessions to a local JSON file and provides
utilities to load and display session history.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

SESSION_FILE = "sessions.json"
MAX_SESSIONS_STORED = 50   # Cap to avoid unbounded file growth


def save_session(routine_name, results):
    """
    Persist a completed session to disk.

    Args:
        routine_name: Display name of the routine (str) or None for free practice.
        results: List of dicts with keys: pose, accuracy, held_for.
    """
    if not results:
        return

    sessions = _load_raw()
    avg_accuracy = round(sum(r["accuracy"] for r in results) / len(results), 1)

    sessions.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "routine": routine_name or "Free Practice",
        "poses_completed": len(results),
        "average_accuracy": avg_accuracy,
        "details": results,
    })

    # Keep only the most recent sessions
    sessions = sessions[-MAX_SESSIONS_STORED:]

    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
        logger.info("Session saved: %s — %d poses, avg %.1f%%", routine_name, len(results), avg_accuracy)
    except OSError as e:
        logger.error("Could not save session: %s", e)


def load_history(max_sessions=5):
    """Return the most recent sessions (newest first)."""
    sessions = _load_raw()
    return list(reversed(sessions[-max_sessions:]))


def format_history_html(max_sessions=5):
    """Return an HTML snippet showing recent session history."""
    sessions = load_history(max_sessions)

    if not sessions:
        return "<p style='color:#95a5a6; text-align:center; font-size:14px;'>No sessions recorded yet.</p>"

    rows = ""
    for s in sessions:
        acc = s.get("average_accuracy", 0)
        color = "#27ae60" if acc >= 80 else "#f39c12" if acc >= 60 else "#e74c3c"
        rows += f"""
        <tr>
            <td style='padding:6px 8px; color:#7f8c8d; font-size:13px;'>{s.get('timestamp', '')}</td>
            <td style='padding:6px 8px; font-weight:600;'>{s.get('routine', '')}</td>
            <td style='padding:6px 8px; text-align:center;'>{s.get('poses_completed', 0)}</td>
            <td style='padding:6px 8px; text-align:center; color:{color}; font-weight:700;'>{acc}%</td>
        </tr>"""

    return f"""
    <div style='overflow-x:auto;'>
      <table style='width:100%; border-collapse:collapse; font-family:Poppins,sans-serif; font-size:14px;'>
        <thead>
          <tr style='border-bottom:2px solid #ecf0f1;'>
            <th style='padding:6px 8px; text-align:left; color:#95a5a6;'>Date</th>
            <th style='padding:6px 8px; text-align:left; color:#95a5a6;'>Routine</th>
            <th style='padding:6px 8px; text-align:center; color:#95a5a6;'>Poses</th>
            <th style='padding:6px 8px; text-align:center; color:#95a5a6;'>Avg Accuracy</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _load_raw():
    if not os.path.exists(SESSION_FILE):
        return []
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load sessions file: %s", e)
        return []
