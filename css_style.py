css = """
/* Overall Body Styling */
body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(to bottom, #A7C7E7, #F2F9FF);
    color: #2C3E50;
    margin: 0;
    padding: 0;
}

/* Container Styling */
.gradio-container {
    background: transparent;
    padding: 10px;
    max-width: 95%;
    margin: 0 auto;
    border-radius: 16px;
}

/* Header Styling */
h1 {
    text-align: center;
    font-size: 2.4em;
    font-weight: bold;
    color: #34495E;
    margin-bottom: 15px;
}

/* Video Stream Styling */
#custom-stream .webrtc-container {
    border: none;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.85);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    padding: 10px;
    margin-bottom: 10px;
}

/* ===== DISTANCE-VISIBLE ACCURACY DISPLAY ===== */
#accuracy-display {
    text-align: center;
    padding: 15px;
    margin: 10px auto;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.95);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    max-width: 400px;
}

#accuracy-display .accuracy-number {
    font-size: 72px;
    font-weight: 900;
    font-family: 'Poppins', sans-serif;
    line-height: 1;
}

#accuracy-display .accuracy-label {
    font-size: 16px;
    color: #7f8c8d;
    margin-top: 4px;
}

/* Color classes for accuracy */
.accuracy-low { color: #e74c3c; }
.accuracy-mid { color: #f39c12; }
.accuracy-high { color: #27ae60; }

/* ===== STATUS DISPLAY ===== */
#status-display {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
    color: #2C3E50;
    padding: 12px 20px;
    margin: 8px auto;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    max-width: 600px;
}

/* ===== FEEDBACK TEXT ===== */
#custom-textbox div {
    font-size: 22px;
    font-family: 'Lora', serif;
    color: #34495E;
    background-color: rgba(255, 255, 255, 0.9);
    border: none;
    border-radius: 16px;
    padding: 16px;
    margin: 0 auto;
    width: 90%;
    max-width: 600px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    text-align: center;
}

/* Audio Feedback Styling */
#feedback-audio {
    width: 90%;
    max-width: 600px;
    border: none;
    background: rgba(173, 216, 230, 0.7);
    border-radius: 12px;
    margin: 10px auto;
    display: block;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Column Layout */
.left-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}

/* ===== MODE SELECTION PANEL ===== */
#mode-panel {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 15px;
}

#mode-panel .gr-button {
    font-size: 18px;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
}

/* ===== PICTOGRAM GRID ===== */
#pictogram-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}

.pictogram img {
    border-radius: 12px;
    opacity: 0.5;
    max-width: 120px;
    transition: all 0.3s ease-in-out;
}

.highlighted img {
    border-radius: 12px;
    opacity: 1.0;
    max-width: 140px;
    border: 3px solid #27ae60;
    box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
    transition: all 0.3s ease-in-out;
}

.completed img {
    border-radius: 12px;
    opacity: 0.7;
    max-width: 120px;
    border: 3px solid #95a5a6;
    filter: grayscale(50%);
}

/* Subtle Transitions */
* {
    transition: all 0.3s ease-in-out;
}

/* ===== RESPONSIVE: PHONE ON GROUND ===== */
@media (max-width: 768px) {
    h1 { font-size: 1.8em; }

    #accuracy-display .accuracy-number {
        font-size: 96px;  /* Even larger on mobile — visible from distance */
    }

    #status-display {
        font-size: 32px;
    }

    #custom-textbox div {
        font-size: 20px;
        padding: 12px;
    }

    .gradio-container {
        max-width: 100%;
        padding: 5px;
    }

    #feedback-audio {
        max-width: 100%;
    }

    /* Stack columns vertically on mobile */
    .gr-row {
        flex-direction: column !important;
    }
}

/* Large screens */
@media (min-width: 1200px) {
    #accuracy-display .accuracy-number {
        font-size: 80px;
    }
}
"""
