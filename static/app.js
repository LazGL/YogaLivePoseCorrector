/**
 * NamastAI — Frontend Application
 *
 * Two WebSocket connections:
 *   /ws/video    — send camera JPEG frames, receive annotated JPEG frames
 *   /ws/feedback — receive real-time JSON (accuracy, status, voice, audio)
 *
 * REST calls:
 *   POST /session/start — start guided or free practice
 *   POST /session/stop  — stop session
 *   GET  /session/history — load past sessions
 *   GET  /config          — load pose/routine definitions
 */

'use strict';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WS_BASE  = `ws://${location.host}`;
const API_BASE = `${location.protocol}//${location.host}`;
const FRAME_INTERVAL_MS = 100;   // Send frames at 10 fps to server
const WS_RECONNECT_MS   = 2000;  // Reconnect delay on disconnect
const JPEG_QUALITY      = 0.75;  // JPEG encoding quality

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let POSES    = {};
let ROUTINES = {};

let videoWs      = null;
let feedbackWs   = null;
let cameraStream = null;
let frameTimer   = null;
let currentAudio = null;
let lastAudioB64 = null;   // Avoid replaying identical audio chunk

const offscreen    = document.createElement('canvas');
const offCtx       = offscreen.getContext('2d');

// ---------------------------------------------------------------------------
// DOM elements
// ---------------------------------------------------------------------------

const canvas          = document.getElementById('output-canvas');
const ctx             = canvas.getContext('2d');
const cameraVideo     = document.getElementById('camera-video');
const accuracyNumber  = document.getElementById('accuracy-number');
const statusDisplay   = document.getElementById('status-display');
const feedbackText    = document.getElementById('feedback-text');
const pictoContainer  = document.getElementById('pictogram-container');
const historyContent  = document.getElementById('history-content');
const cameraStatus    = document.getElementById('camera-status');

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

async function init() {
    try {
        const cfg = await fetch(`${API_BASE}/config`).then(r => r.json());
        POSES    = cfg.poses    || {};
        ROUTINES = cfg.routines || {};
    } catch (e) {
        console.error('Failed to load config:', e);
    }

    buildModeButtons();
    renderPictograms(null, []);
    connectFeedbackWS();
    await startCamera();
    connectVideoWS();
    loadHistory();
}

// ---------------------------------------------------------------------------
// Mode buttons (built dynamically from server config)
// ---------------------------------------------------------------------------

function buildModeButtons() {
    const guidedDiv = document.getElementById('guided-buttons');
    const freeDiv   = document.getElementById('free-buttons');

    Object.entries(ROUTINES).forEach(([key, routine]) => {
        const btn = document.createElement('button');
        btn.className   = 'btn-primary';
        btn.textContent = routine.name;
        btn.onclick     = () => startSession('guided', key);
        guidedDiv.appendChild(btn);
    });

    Object.entries(POSES).forEach(([key, pose]) => {
        const btn = document.createElement('button');
        btn.className   = 'btn-secondary';
        btn.textContent = pose.name;
        btn.onclick     = () => startSession('free', key);
        freeDiv.appendChild(btn);
    });
}

// ---------------------------------------------------------------------------
// Pictograms
// ---------------------------------------------------------------------------

function renderPictograms(activePose, completedPoses) {
    pictoContainer.innerHTML = '';

    Object.entries(POSES).forEach(([key, pose]) => {
        if (!pose.pictogram_url) return;

        const isActive    = key === activePose;
        const isCompleted = completedPoses.includes(key);

        const wrap = document.createElement('div');
        wrap.style.textAlign = 'center';

        const img = document.createElement('img');
        img.src = pose.pictogram_url;
        img.alt = pose.name;

        if (isActive) {
            img.style.cssText = [
                'border-radius:12px',
                'border:3px solid #27ae60',
                'opacity:1.0',
                'max-width:140px',
                'box-shadow:0 4px 12px rgba(39,174,96,0.3)',
                'transition:all 0.3s',
            ].join(';');
        } else if (isCompleted) {
            img.style.cssText = [
                'border-radius:12px',
                'border:3px solid #95a5a6',
                'opacity:0.7',
                'max-width:120px',
                'filter:grayscale(50%)',
                'transition:all 0.3s',
            ].join(';');
        } else {
            img.style.cssText = [
                'border-radius:12px',
                'border:2px solid transparent',
                'opacity:0.5',
                'max-width:120px',
                'transition:all 0.3s',
            ].join(';');
        }

        const label = document.createElement('div');
        label.style.cssText = `font-size:13px; font-weight:600; margin-top:4px; color:${
            isActive ? '#27ae60' : isCompleted ? '#95a5a6' : '#7f8c8d'
        }`;
        label.textContent = pose.name;

        wrap.appendChild(img);
        wrap.appendChild(label);
        pictoContainer.appendChild(wrap);
    });
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        });
        cameraVideo.srcObject = cameraStream;
        await new Promise(resolve => { cameraVideo.onloadedmetadata = resolve; });

        // Match canvas to actual camera resolution
        canvas.width      = cameraVideo.videoWidth  || 640;
        canvas.height     = cameraVideo.videoHeight || 480;
        offscreen.width   = canvas.width;
        offscreen.height  = canvas.height;

        cameraStatus.style.display = 'none';
    } catch (err) {
        console.error('Camera error:', err);
        cameraStatus.style.display = 'block';
    }
}

// ---------------------------------------------------------------------------
// Video WebSocket — sends camera frames, receives annotated frames
// ---------------------------------------------------------------------------

function connectVideoWS() {
    if (videoWs) { videoWs.onclose = null; videoWs.close(); }

    videoWs = new WebSocket(`${WS_BASE}/ws/video`);
    videoWs.binaryType = 'arraybuffer';

    videoWs.onopen = () => {
        console.log('[video WS] connected');
        startSendingFrames();
    };

    videoWs.onmessage = (event) => {
        // Display annotated frame on canvas
        const blob = new Blob([event.data], { type: 'image/jpeg' });
        const url  = URL.createObjectURL(blob);
        const img  = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(url);
        };
        img.src = url;
    };

    videoWs.onclose = () => {
        console.warn('[video WS] disconnected — reconnecting in', WS_RECONNECT_MS, 'ms');
        stopSendingFrames();
        setTimeout(connectVideoWS, WS_RECONNECT_MS);
    };

    videoWs.onerror = (e) => console.error('[video WS] error:', e);
}

function startSendingFrames() {
    stopSendingFrames();
    frameTimer = setInterval(() => {
        if (!cameraStream || videoWs?.readyState !== WebSocket.OPEN) return;

        offCtx.drawImage(cameraVideo, 0, 0, offscreen.width, offscreen.height);
        offscreen.toBlob(blob => {
            if (blob && videoWs?.readyState === WebSocket.OPEN) {
                blob.arrayBuffer().then(buf => videoWs.send(buf));
            }
        }, 'image/jpeg', JPEG_QUALITY);
    }, FRAME_INTERVAL_MS);
}

function stopSendingFrames() {
    if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
}

// ---------------------------------------------------------------------------
// Feedback WebSocket — receives real-time JSON and updates UI
// ---------------------------------------------------------------------------

function connectFeedbackWS() {
    if (feedbackWs) { feedbackWs.onclose = null; feedbackWs.close(); }

    feedbackWs = new WebSocket(`${WS_BASE}/ws/feedback`);

    feedbackWs.onmessage = (event) => {
        try {
            updateUI(JSON.parse(event.data));
        } catch (e) {
            console.error('[feedback WS] parse error:', e);
        }
    };

    feedbackWs.onclose = () => {
        console.warn('[feedback WS] disconnected — reconnecting in', WS_RECONNECT_MS, 'ms');
        setTimeout(connectFeedbackWS, WS_RECONNECT_MS);
    };

    feedbackWs.onerror = (e) => console.error('[feedback WS] error:', e);
}

function updateUI(data) {
    // Accuracy number + colour
    const acc = Math.round(data.accuracy || 0);
    accuracyNumber.textContent = `${acc}%`;
    accuracyNumber.className   = 'accuracy-number ' + (
        acc >= 80 ? 'accuracy-high' : acc >= 60 ? 'accuracy-mid' : 'accuracy-low'
    );

    // Status line
    statusDisplay.textContent = data.status || '';

    // Feedback text
    feedbackText.textContent = data.feedback || '';

    // Pictograms (only re-render when something changed)
    renderPictograms(data.active_pose || null, data.completed_poses || []);

    // Audio playback (server sends base64-encoded MP3/WAV only when new audio is ready)
    if (data.audio_b64 && data.audio_b64 !== lastAudioB64) {
        lastAudioB64 = data.audio_b64;
        playAudio(data.audio_b64, data.audio_type || 'mp3');
    }
}

function playAudio(b64, type) {
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    const mime = type === 'wav' ? 'audio/wav' : 'audio/mpeg';
    currentAudio = new Audio(`data:${mime};base64,${b64}`);
    currentAudio.play().catch(e => console.warn('Audio play blocked:', e));
}

// ---------------------------------------------------------------------------
// Session control
// ---------------------------------------------------------------------------

async function startSession(mode, key) {
    try {
        const resp = await fetch(`${API_BASE}/session/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, key }),
        });
        const data = await resp.json();
        statusDisplay.textContent = data.status || data.error || '';
    } catch (e) {
        console.error('startSession error:', e);
    }
}

async function stopSession() {
    try {
        const resp = await fetch(`${API_BASE}/session/stop`, { method: 'POST' });
        const data = await resp.json();
        statusDisplay.textContent = data.status || '';
    } catch (e) {
        console.error('stopSession error:', e);
    }
}

// ---------------------------------------------------------------------------
// Session history
// ---------------------------------------------------------------------------

async function loadHistory() {
    try {
        const sessions = await fetch(`${API_BASE}/session/history`).then(r => r.json());
        renderHistory(sessions);
    } catch (e) {
        console.error('loadHistory error:', e);
    }
}

function renderHistory(sessions) {
    if (!sessions || sessions.length === 0) {
        historyContent.innerHTML =
            "<p style='color:#95a5a6; text-align:center; font-size:14px;'>No sessions yet.</p>";
        return;
    }

    const rows = sessions.map(s => {
        const acc   = s.average_accuracy || 0;
        const color = acc >= 80 ? '#27ae60' : acc >= 60 ? '#f39c12' : '#e74c3c';
        return `
        <tr>
            <td style="padding:6px 8px; color:#7f8c8d; font-size:13px;">${s.timestamp || ''}</td>
            <td style="padding:6px 8px; font-weight:600;">${s.routine || ''}</td>
            <td style="padding:6px 8px; text-align:center;">${s.poses_completed || 0}</td>
            <td style="padding:6px 8px; text-align:center; color:${color}; font-weight:700;">${acc}%</td>
        </tr>`;
    }).join('');

    historyContent.innerHTML = `
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <thead>
            <tr style="border-bottom:2px solid #ecf0f1;">
                <th style="padding:6px 8px; text-align:left; color:#95a5a6;">Date</th>
                <th style="padding:6px 8px; text-align:left; color:#95a5a6;">Routine</th>
                <th style="padding:6px 8px; text-align:center; color:#95a5a6;">Poses</th>
                <th style="padding:6px 8px; text-align:center; color:#95a5a6;">Avg Acc</th>
            </tr>
        </thead>
        <tbody>${rows}</tbody>
    </table>`;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

window.addEventListener('load', init);
