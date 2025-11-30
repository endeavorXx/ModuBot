"""
Configuration file for NAO Brain System
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
EMBEDDING_FILE = "./src/my_nao_controller/my_nao_controller/action_embeddings.pkl"

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
ROS_NODE_NAME = "nao_brain"
ROS_ACTION_TOPIC = "/perform_action"
ROS_QUEUE_SIZE = 10

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Verbosity levels
VERBOSE_LOGGING = True  # Detailed logs for debugging
LOG_TIMESTAMPS = True   # Include timestamps in logs
LOG_LATENCY = True      # Log processing latency for each component

# Log formatting
LOG_SEPARATOR = "=" * 50

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================
# Threading
ENABLE_PARALLEL_EXECUTION = True  # Run TTS and action search in parallel

# Timeouts (seconds)
STT_TIMEOUT = 30
TTS_TIMEOUT = 30
LLM_TIMEOUT = 60

# ============================================================================
# ERROR HANDLING CONFIGURATION
# ============================================================================
# Fallback responses
FALLBACK_TRANSCRIPT = "..."
FALLBACK_INTENT = "neutral"
FALLBACK_SPEECH = "I cannot connect"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Seconds between retries

# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================
# Model loading
FORCE_MODEL_RELOAD = False  # Force reload of ML models on startup
CACHE_EMBEDDINGS = True     # Cache embeddings for faster lookups

# Audio preprocessing
NORMALIZE_AUDIO = True      # Normalize audio before saving
AUDIO_FORMAT = "int16"      # Audio sample format

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary():
    """Returns a formatted summary of current configuration"""
    return f"""
NAO Brain Configuration Summary
{LOG_SEPARATOR}
Persona: {CURRENT_PERSONA_ID}
LLM Model: {LLM_MODEL}
Memory: {MAX_CONVERSATION_HISTORY} exchanges
Action Threshold: {ACTION_CONFIDENCE_THRESHOLD}
Parallel Execution: {ENABLE_PARALLEL_EXECUTION}
{LOG_SEPARATOR}
"""

def validate_config():
    """Validate critical configuration parameters"""
    errors = []
    
    # Check API credentials
    if not NVIDIA_API_KEY or NVIDIA_API_KEY.startswith("nvapi-xxx"):
        errors.append("NVIDIA_API_KEY not set properly")
    
    if not STT_FUNCTION_ID or STT_FUNCTION_ID == "your-stt-function-id":
        errors.append("STT_FUNCTION_ID not set properly")
    
    if not TTS_FUNCTION_ID or TTS_FUNCTION_ID == "your-tts-function-id":
        errors.append("TTS_FUNCTION_ID not set properly")
    
    # Check file paths
    if not os.path.exists(CLIENTS_REPO_PATH):
        errors.append(f"CLIENTS_REPO_PATH not found: {CLIENTS_REPO_PATH}")
    
    # Check numeric ranges
    if MAX_CONVERSATION_HISTORY < 1:
        errors.append("MAX_CONVERSATION_HISTORY must be at least 1")
    
    if ACTION_CONFIDENCE_THRESHOLD < 0 or ACTION_CONFIDENCE_THRESHOLD > 1:
        errors.append("ACTION_CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# ============================================================================
# ENVIRONMENT VARIABLE SUPPORT (Optional)
# ============================================================================
# Uncomment these lines if you want to override with environment variables

# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY)
# STT_FUNCTION_ID = os.getenv("STT_FUNCTION_ID", STT_FUNCTION_ID)
# TTS_FUNCTION_ID = os.getenv("TTS_FUNCTION_ID", TTS_FUNCTION_ID)
# CURRENT_PERSONA_ID = os.getenv("CURRENT_PERSONA_ID", CURRENT_PERSONA_ID)