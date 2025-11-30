"""
nao_action_vocab.py - Action Vocabulary for NAO Robot Gestures

This module defines all available actions (gestures/animations) that the
NAO robot can perform. Actions are organized into two categories:

1. NAO_ACTIONS (dict): Static poses - instant position changes
   Currently empty as all actions use the complex animation system.

2. NAO_COMPLEX_ACTIONS (dict): Time-based animations using sine wave
   interpolation for smooth, natural movements.

Each complex action contains:
    - description (str): Natural language description for semantic search
    - duration (float): Total animation time in seconds
    - repetitions (int): Number of sine wave cycles
    - curves (list): Joint movement definitions with min/max positions

The 'description' field is used by the ActionSelector to match LLM intents
to appropriate robot gestures via embedding similarity search.

Author: Vashu Chauhan
"""

# Static poses dictionary - instant position changes
# Currently empty as all actions are implemented as complex animations
NAO_ACTIONS = {
    
}

NAO_COMPLEX_ACTIONS = {
    
    # 1. Stand Neutral (Moved from Static)
    "stand_neutral": {
        "description": "Resets the robot to a standard standing posture with arms by the side and head straight.",
        "duration": 2.0,
        "repetitions": 1,
        "curves": [
            # We set min and max to the same value to hold the position
            { "joint": "RShoulderPitch", "min": 1.5, "max": 1.5 },
            { "joint": "LShoulderPitch", "min": 1.5, "max": 1.5 },
            { "joint": "RShoulderRoll", "min": -0.1, "max": -0.1 },
            { "joint": "LShoulderRoll", "min": 0.1, "max": 0.1 },
            { "joint": "HeadPitch", "min": 0.0, "max": 0.0 },
            { "joint": "HeadYaw", "min": 0.0, "max": 0.0 },
            { "joint": "RElbowRoll", "min": 0.0, "max": 0.0 },
            { "joint": "LElbowRoll", "min": 0.0, "max": 0.0 },
            { "joint": "LPhalanx1", "min": 0.0, "max": 0.0 },
            { "joint": "LPhalanx2", "min": 0.0, "max": 0.0 },
            { "joint": "LPhalanx3", "min": 0.0, "max": 0.0 }
        ]
    },

    # 2. Shake Head
    "shake_head": {
        "description": "Rotates head left and right continuously to indicate disagreement or saying no.",
        "duration": 3.0,
        "repetitions": 3,
        "curves": [
            { "joint": "HeadYaw", "min": -1.5, "max": 1.5 }
        ]
    },

    # 3. Nod Head
    "nod_head": {
        "description": "Moves head up and down continuously to indicate agreement or saying yes.",
        "duration": 3.0,
        "repetitions": 3,
        "curves": [
            { "joint": "HeadPitch", "min": -0.5, "max": 0.3 }
        ]
    },
    
    # 4. Get Out (Left Hand)
    "get_out_left_hand": {
        "description": "Aggressive gesture raising the left arm outward and pointing to the side, acting like asking someone to leave.",
        "duration": 4.0,
        "repetitions": 1,
        "curves": [
            # 1. Arm moves UP/FORWARD (1.5 -> 0.0 -> 1.5)
            # # we are using sin wave function to create smooth motion, Now if you want to freeze at max position first set repetitions to 0.5
            { "joint": "LShoulderPitch", "min": 0.0, "max": 1.5, "start_from_max": True },
            # 2. Arm moves OUT (0.0 -> 1.2 -> 0.0)
            { "joint": "LShoulderRoll", "min": 0.0, "max": 1.2 },
            # 3. Fingers OPEN (0.0 -> 1.0 -> 0.0)
            { "joint": "LPhalanx1", "min": 0.0, "max": 1.0 },
            { "joint": "LPhalanx2", "min": 0.0, "max": 1.0 },
            { "joint": "LPhalanx3", "min": 0.0, "max": 1.0 },
            
        ]
    },
    "get_out_right_hand": {
        "description": "Aggressive gesture raising the right arm outward and pointing to the side, acting like asking someone to leave.",
        "duration": 4.0,
        "repetitions": 1,
        "curves": [
            { "joint": "RShoulderPitch", "min": 0.0, "max": 1.5, "start_from_max": True },
            { "joint": "RShoulderRoll", "min": -1.3, "max": 0, "start_from_max": True },
            { "joint": "RPhalanx4", "min": 0.0, "max": 1.0 },
            { "joint": "RPhalanx5", "min": 0.0, "max": 1.0 },
            { "joint": "RPhalanx6", "min": 0.0, "max": 1.0 },
        ]
    },
    "wave_right_hand": {
        "description": "Raises right arm and waves the hand back and forth. Friendly greeting.",
        "duration": 6.0,
        "repetitions": 2,
        "curves": [
            { "joint": "RShoulderPitch", "min": -1.0, "max": -1.0 }, # Arm Up
            { "joint": "RShoulderRoll", "min": -0.3, "max": -0.3 },  # Arm slightly out
            { "joint": "RElbowRoll", "min": 0.5, "max": 1.2 },       # Waving motion
            { "joint": "RPhalanx1", "min": 0.0, "max": 1.0 },        # Open hand
            { "joint": "RPhalanx2", "min": 0.0, "max": 1.0 },
            { "joint": "RPhalanx3", "min": 0.0, "max": 1.0 },
        ]
    },

    # ---------------------------------------------------------
    # TEACHING & PRESENTATION GESTURES
    # ---------------------------------------------------------

    # 8. Explain Open Arms (Teaching)
    "explain_open_arms": {
        "description": "Spreads both arms wide open. Used when explaining a big concept or welcoming.",
        "duration": 5.0,
        "repetitions": 1,
        "curves": [
            { "joint": "RShoulderPitch", "min": 0.5, "max": 0.5 },
            { "joint": "LShoulderPitch", "min": 0.5, "max": 0.5 },
            { "joint": "RShoulderRoll", "min": -0.2, "max": -1.0 }, # Open wide
            { "joint": "LShoulderRoll", "min": 0.2, "max": 1.0 },   # Open wide
            { "joint": "RElbowRoll", "min": 0.5, "max": 0.5 },      # Slight natural bend
            { "joint": "LElbowRoll", "min": -0.5, "max": -0.5 },
            { "joint": "LPhalanx1", "min": 1.0, "max": 1.0 },       # Hands Open
        ]
    },

    # 22. Hands on Hips (Power Pose)
    "hands_on_hips": {
        "description": "Places fists on hips. Authoritative or scolding.",
        "duration": 4.0,
        "repetitions": 1,
        "curves": [
            { "joint": "RShoulderPitch", "min": 1.0, "max": 1.0 },
            { "joint": "LShoulderPitch", "min": 1.0, "max": 1.0 },
            { "joint": "RShoulderRoll", "min": -1.2, "max": -1.2 }, # Elbows out
            { "joint": "LShoulderRoll", "min": 1.2, "max": 1.2 },   # Elbows out
            { "joint": "RElbowRoll", "min": 1.5, "max": 1.5 },      # Bend
            { "joint": "LElbowRoll", "min": -1.5, "max": -1.5 },    # Bend
            { "joint": "RElbowYaw", "min": 0.5, "max": 0.5 },       # Knuckles to hip
            { "joint": "LElbowYaw", "min": -0.5, "max": -0.5 },
        ]
    }
}