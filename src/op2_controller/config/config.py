"""
Configuration file for OP2 Brain System
Place this file in: ../config/config.py (relative to the script)
"""

import os
from pathlib import Path

# ============================================================================
# API CREDENTIALS
# ============================================================================
# NVIDIA API Configuration is present in the .env file (create your own from .env.template)
# ============================================================================

NVIDIA_API_KEY="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"
STT_FUNCTION_ID="d3fe9151-442b-4204-a70d-5fcc597fd610"
TTS_FUNCTION_ID="877104f7-e885-42b9-8de8-f6e4c6303969"
# PERSONA CONFIGURATION
# ============================================================================
# Options: "polite_teacher", "polite_receptionist", "angry_cab_driver"
CURRENT_PERSONA_ID = "angry_cab_driver"

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Base paths (relative to script location)
CLIENTS_REPO_PATH = "./python-clients"
EMBEDDING_FILE = "./src/op2_controller/op2_controller/action_embeddings.pkl"

# Script paths
STT_SCRIPT = os.path.join(CLIENTS_REPO_PATH, "scripts/asr/transcribe_file_offline.py")
TTS_SCRIPT = os.path.join(CLIENTS_REPO_PATH, "scripts/tts/talk.py")

# Audio file paths
os.makedirs("assets", exist_ok=True)
AUDIO_INPUT = "assets/user_input.wav"
AUDIO_OUTPUT = "assets/robot_response.wav"

# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================
# Audio recording settings
SAMPLE_RATE = 16000  # Hz
BLOCK_SIZE = 512     # Audio buffer block size
CHANNELS = 1         # Mono audio

# Voice Activity Detection (VAD)
SILENCE_THRESHOLD = 2.0  # Seconds of silence before stopping recording
VAD_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for speech detection

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================
RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
USE_SSL = True

# ============================================================================
# SPEECH-TO-TEXT (STT) CONFIGURATION
# ============================================================================
STT_LANGUAGE_CODE = "en-US"
STT_MODEL = "parakeet"  # Model identifier for logging

# ============================================================================
# TEXT-TO-SPEECH (TTS) CONFIGURATION
# ============================================================================
# Default TTS settings (can be overridden by persona)
TTS_LANGUAGE_CODE = "EN-US"  # Note: Uppercase for TTS API
TTS_VOICE = "Magpie-Multilingual.EN-US.Aria"
TTS_MODEL = "magpie"  # Model identifier for logging

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
# Model settings
LLM_MODEL = "meta/llama3-8b-instruct"
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Generation parameters
LLM_TEMPERATURE = 0.7  # Higher = more creative/varied responses
LLM_MAX_TOKENS = 128   # Maximum tokens in response
LLM_STREAMING = True   # Enable streaming responses

# ============================================================================
# MEMORY CONFIGURATION
# ============================================================================
# Conversation history settings
MAX_CONVERSATION_HISTORY = 20  # Number of exchanges to keep (each exchange = user + assistant)
ENABLE_CONVERSATION_MEMORY = True  # Set to False to disable memory

# ============================================================================
# ACTION SELECTION CONFIGURATION
# ============================================================================
# Semantic search settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ACTION_CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence for action mapping
DEFAULT_ACTION = "stand_neutral"   # Fallback action for low confidence

# ============================================================================
# ROS2 CONFIGURATION
# ============================================================================
