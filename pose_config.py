"""
Pose definitions and workout routines for NamastAI.

Each pose has a reference image, a tag (standing/ground) for measurement filtering,
a display name, a pictogram for the UI, and a default hold duration.
"""

POSES = {
    "tpose": {
        "name": "T-Pose",
        "reference_image": "Tpose.png",
        "pictogram": "images_front_end/Tpose_picto.png",
        "tag": "standing",
        "hold_seconds": 20,
    },
    "warrior2": {
        "name": "Warrior II",
        "reference_image": "target2.png",
        "pictogram": "images_front_end/Warrior2_picto.png",
        "tag": "standing",
        "hold_seconds": 30,
    },
    "warrior2_handsup": {
        "name": "Warrior II Arms Up",
        "reference_image": "target2.png",  # TODO: add dedicated reference image
        "pictogram": "images_front_end/Warrior2_handsup_picto.png",
        "tag": "standing",
        "hold_seconds": 30,
    },
    "squat": {
        "name": "Squat",
        "reference_image": "target2.png",  # TODO: add dedicated reference image
        "pictogram": "images_front_end/Squat_picto.png",
        "tag": "standing",
        "hold_seconds": 25,
    },
}

ROUTINES = {
    "beginner_flow": {
        "name": "Beginner Flow",
        "description": "A gentle introduction to basic yoga poses",
        "poses": ["tpose", "warrior2", "squat"],
        "rest_seconds": 10,
    },
    "warrior_series": {
        "name": "Warrior Series",
        "description": "Build strength with warrior poses",
        "poses": ["warrior2", "warrior2_handsup", "warrior2"],
        "rest_seconds": 10,
    },
    "full_program": {
        "name": "Full Program",
        "description": "Complete all available poses",
        "poses": ["tpose", "warrior2", "warrior2_handsup", "squat"],
        "rest_seconds": 10,
    },
}

# Accuracy thresholds
ACCURACY_PERFECT = 80   # Pose is held correctly
ACCURACY_FEEDBACK = 60  # Needs correction feedback
# Below ACCURACY_FEEDBACK: pose not detected or too far off
