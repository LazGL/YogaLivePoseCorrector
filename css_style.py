css = """
/* Overall Body Styling */
body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(to bottom, #A7C7E7, #F2F9FF); /* Soothing blue gradient */
    color: #2C3E50; /* Deep calming blue for text */
    margin: 0;
    padding: 0;
}

/* Container Styling */
.gradio-container {
    background: transparent;
    padding: 20px;
    max-width: 90%;
    margin: 0 auto;
    border-radius: 16px;
}

/* Header Styling */
h1 {
    text-align: center;
    font-size: 2.8em;
    font-weight: bold;
    color: #34495E;
    margin-bottom: 25px;
    letter-spacing: 1px;
}

/* Video Stream Styling */
#custom-stream .webrtc-container {
    border: none;
    border-radius: 20px; /* Rounded corners for calmness */
    background: rgba(255, 255, 255, 0.85); /* Soft white overlay */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); /* Deeper shadow for focus */
    padding: 15px;
    margin-bottom: 20px;
}

/* Feedback Text Styling */
#custom-textbox div {
    font-size: 20px;
    font-family: 'Lora', serif; /* Elegant and calming serif font */
    color: #34495E; /* Darker, grounding text color */
    background-color: rgba(255, 255, 255, 0.9); /* Subtle white overlay */
    border: none; /* Removed unnecessary border */
    border-radius: 20px;
    padding: 20px;
    margin: 0 auto;
    width: 90%; /* Proportional width */
    max-width: 600px; /* Restrict to align with the layout */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Light shadow for depth */
    text-align: center;
}

/* Audio Feedback Styling */
#feedback-audio {
    width: 90%;
    max-width: 600px; /* Restrict width for alignment */
    border: none;
    background: rgba(173, 216, 230, 0.7); /* Gentle blue background */
    border-radius: 12px;
    margin: 15px auto;
    display: block;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow */
}

/* Align Left Column */
.left-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px; /* Consistent spacing */
}

/* Align Right Column */
.right-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

/* Image Styling */
#pose-image img {
    border-radius: 16px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Depth for pose image */
    max-width: 100%;
}

/* Subtle Transitions for All Elements */
* {
    transition: all 0.3s ease-in-out;
}

/* Responsive Design */
@media (max-width: 768px) {
    h1 {
        font-size: 2em;
    }
    #custom-textbox div {
        font-size: 18px;
        padding: 15px;
    }
    #feedback-audio {
        max-width: 100%;
    }
    .my-group {
        width: 90%;
        padding: 15px;
    }
}
"""