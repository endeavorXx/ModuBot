"""
op2_action_vocab.py - Action Vocabulary for Robotis OP2 Robot Gestures

This module defines all available actions (gestures/animations) that the
Robotis OP2 robot can perform. Actions are organized into two categories:

1. OP2_ACTIONS (dict): Static poses - instant position changes
   Currently empty as all actions use the complex animation system.

2. OP2_COMPLEX_ACTIONS (dict): Time-based animations using sine wave
   interpolation for smooth, natural movements.

Each complex action contains:
    - description (str): Natural language description for semantic search
    - duration (float): Total animation time in seconds
    - repetitions (int): Number of sine wave cycles
    - curves (list): Joint movement definitions with min/max positions

The 'description' field is used by the ActionSelector to match LLM intents
to appropriate robot gestures via embedding similarity search.
"""

# Static poses dictionary - instant position changes
OP2_ACTIONS = {
    
}

OP2_COMPLEX_ACTIONS = {
    
    # 1. Stand Neutral
    "stand_neutral": {
        "description": "Resets the robot to a standard standing posture with arms by the side and head straight.",
        "duration": 2.0,
        "repetitions": 1,
        "curves": [
            # Arms down
            { "joint": "ShoulderR", "min": -0.25, "max": -0.25 },
            { "joint": "ShoulderL", "min": 0.25, "max": 0.25 },
            { "joint": "ArmUpperR", "min": -0.6, "max": -0.5, "start_from_max" : True},
            { "joint": "ArmUpperL", "min": 0.2, "max": 0.6 },
            { "joint": "ArmLowerR", "min": -0.5, "max": -0.5 },
            { "joint": "ArmLowerL", "min": 0.5, "max": 0.5 },
            # Head straight
            { "joint": "Neck", "min": 0.0, "max": 0.0 },
            { "joint": "Head", "min": 0.0, "max": 0.2 },
        ]
    },

    # 2. Wave Right Hand
    "wave_right_hand": {
        "description": "Raises right arm and waves hand to greet someone or say hello.",
        "duration": 3.0,
        "repetitions": 3,
        "curves": [
            # Raise arm
            { "joint": "ShoulderR", "min": -0.5, "max": 2.0 }, # Up
            { "joint": "ArmUpperR", "min": 0.0, "max": 0.0 }, # Out
            # Wave elbow
            { "joint": "ArmLowerR", "min": -1.0, "max": 1.0 },
        ]
    },

    # 3. Nod Head
    "nod_head": {
        "description": "Nods head up and down to show agreement or understanding.",
        "duration": 2.0,
        "repetitions": 2,
        "curves": [
            { "joint": "Head", "min": -0.3, "max": 0.3 },
        ]
    },
    
    # 4. Shake Head
    "shake_head": {
        "description": "Shakes head left and right to show disagreement or refusal.",
        "duration": 2.0,
        "repetitions": 2,
        "curves": [
            { "joint": "Neck", "min": -0.75, "max": 0.75 },
        ]
    }
}
