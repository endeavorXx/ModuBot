# Nao Robot Persona-Aware Controller

A ROS 2 package that controls a simulated Nao humanoid robot in Webots. The robot uses **LLM-powered conversation** with **persona-aware behavior**, performing context-appropriate gestures and animations based on semantic intent matching.

---

## Table of Contents
- [System Requirements](#system-requirements)
- [Directory Structure](#directory-structure)
- [Architecture Overview](#architecture-overview)
- [File Descriptions](#file-descriptions)
- [Configuration Reference](#configuration-reference)
- [Installation](#installation)
- [Building the Project](#building-the-project)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS (Jammy Jellyfish) |
| ROS 2 | Humble Hawksbill |
| Simulator | Webots R2025a |
| Python | 3.10+ |

---

## Directory Structure

```
~/ros2_ws/src/my_nao_controller/
├── package.xml                 # ROS 2 package manifest
├── setup.py                    # Python package setup & entry points
├── setup.cfg                   # Package metadata
├── notes.txt                   # Development notes
│
├── config/
│   └── config.py               # Central configuration (API keys, paths, parameters)
│
├── docs/
│   ├── README.md
├── launch/
│   └── robot_launch.py         # ROS 2 launch file for Webots + driver
│
├── my_nao_controller/          # Main Python package
│   ├── __init__.py
│   ├── nao_driver.py           # Animation engine (Webots controller)
│   ├── nao_brain.py            # Main AI node (STT → LLM → TTS → Action)
│   ├── nao_action_vocab.py     # Action vocabulary definitions
│   ├── personas.py             # Persona definitions (personality profiles)
│   ├── generate_action_embeddings.py  # Pre-compute action embeddings
│   ├── test_nao_actions.py     # Manual action tester
│   └── action_embeddings.pkl   # Pre-computed embeddings (generated)
│
├── resource/
│   ├── my_nao_controller       # Ament resource marker
│   └── nao.urdf                # Robot description linking to Python driver
└── worlds/
    └── nao_world.wbt           # Webots simulation world file
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NAO BRAIN (nao_brain.py)                       │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────────┐             │
│  │   VAD   │───▶│   STT   │───▶│   LLM   │───▶│ ActionSelect │             │
│  │ (Silero)│    │ (Riva)  │    │ (Llama) │    │ (Embeddings) │             │
│  └─────────┘    └─────────┘    └─────────┘    └──────┬───────┘             │
│       ▲                              │               │                      │
│       │                              ▼               ▼                      │
│   [Microphone]                 ┌─────────┐    ┌─────────────┐              │
│                                │   TTS   │    │  Publish to │              │
│                                │ (Riva)  │    │/perform_action│             │
│                                └────┬────┘    └──────┬──────┘              │
│                                     │                │                      │
│                                     ▼                │                      │
│                               [Speaker]              │                      │
└──────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                           ROS 2 Topic: /perform_action│
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NAO DRIVER (nao_driver.py)                        │
│                                                                             │
│  ┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ Action Callback│───▶│ Animation Engine│───▶│  Webots Motors  │          │
│  │ (ROS Subscriber)│   │ (Sine Wave Math)│    │ (Joint Control) │          │
│  └────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                             │
│  Action Types:                                                              │
│  • Static: Instant pose (set position once)                                 │
│  • Complex: Time-based animation (sine wave interpolation)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │     WEBOTS      │
                              │  NAO Simulator  │
                              └─────────────────┘
```

### Data Flow

1. **Voice Input**: User speaks → VAD detects speech → Audio recorded
2. **Speech-to-Text**: Nvidia Riva transcribes audio to text
3. **LLM Processing**: Persona-aware prompt + conversation history → generates `intent | speech`
4. **Action Selection**: Semantic search matches intent to closest action using embeddings
5. **Text-to-Speech**: Nvidia Riva synthesizes speech audio
6. **Parallel Execution**: Action published to robot + audio played (concurrent)
7. **Animation**: Driver receives action → executes sine-wave animation on joints

---

## File Descriptions

### Core Files

| File | Purpose |
|------|---------|
| `nao_brain.py` | **Main AI node**. Runs the conversation loop: listens for speech (VAD), transcribes (STT), generates persona-aware responses (LLM), maps intent to action (semantic search), synthesizes speech (TTS), and publishes actions. Supports conversation memory. |
| `nao_driver.py` | **Animation engine**. A Webots controller that subscribes to `/perform_action` and animates the robot. Uses sine-wave interpolation for smooth complex motions. |
| `nao_action_vocab.py` | **Action vocabulary**. Defines `NAO_ACTIONS` (static poses) and `NAO_COMPLEX_ACTIONS` (timed animations with joint curves and descriptions for embedding). |
| `personas.py` | **Persona definitions**. Contains personality profiles (e.g., `polite_teacher`, `angry_cab_driver`) with speaking style, example phrases, and TTS voice settings. |
| `generate_action_embeddings.py` | **Embedding generator**. Pre-computes sentence embeddings for action descriptions using `all-MiniLM-L6-v2`. Outputs `action_embeddings.pkl`. |
| `test_nao_actions.py` | **Manual tester**. Publishes action commands sequentially to test robot movements without the full AI pipeline. |

### Configuration

| File | Purpose |
|------|---------|
| `config/config.py` | **Central configuration**. All tunable parameters: API keys, model settings, paths, audio config, timeouts, thresholds, ROS settings, and logging options. |

### Launch & Resources

| File | Purpose |
|------|---------|
| `launch/robot_launch.py` | **ROS 2 launch file**. Starts Webots with the world file and attaches the `NaoDriver` controller. |
| `resource/nao.urdf` | **Robot description**. Links the Webots robot to the Python driver class via `<plugin type="...">`. |
| `worlds/nao_world.wbt` | **Webots world**. Contains the Nao robot with `controller "<extern>"` to use external ROS control. |

---

## Configuration Reference

All configuration is centralized in `config/config.py`:

### API Credentials

```python
NVIDIA_API_KEY = "nvapi-..."       # Nvidia NIM API key
STT_FUNCTION_ID = "..."            # Speech-to-Text function ID
TTS_FUNCTION_ID = "..."            # Text-to-Speech function ID
```

### Persona Selection

```python
CURRENT_PERSONA_ID = "angry_cab_driver"  # Options: polite_teacher, polite_receptionist, angry_cab_driver
```

### Path Configuration

```python
CLIENTS_REPO_PATH = "./python-clients"                              # Nvidia Riva client scripts
EMBEDDING_FILE = "./src/my_nao_controller/.../action_embeddings.pkl"
AUDIO_INPUT = "assets/user_input.wav"
AUDIO_OUTPUT = "assets/robot_response.wav"
```

### Audio Settings

```python
SAMPLE_RATE = 16000          # Hz
SILENCE_THRESHOLD = 2.0      # Seconds of silence before stopping
VAD_CONFIDENCE_THRESHOLD = 0.5
```

### LLM Settings

```python
LLM_MODEL = "meta/llama3-8b-instruct"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 128
```

### Memory & Actions

```python
ENABLE_CONVERSATION_MEMORY = True
MAX_CONVERSATION_HISTORY = 20        # Exchanges to remember
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ACTION_CONFIDENCE_THRESHOLD = 0.1    # Below this → fallback to stand_neutral
DEFAULT_ACTION = "stand_neutral"
```

### Performance

```python
ENABLE_PARALLEL_EXECUTION = True     # TTS + action search in parallel
STT_TIMEOUT = 30
TTS_TIMEOUT = 30
```

---

## Installation

### Step 1: Install ROS 2 Humble

```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop ros-dev-tools
```

### Step 2: Install Webots R2025a

> ⚠️ Do NOT use `apt install webots` — it may install an older version.

```bash
cd ~/Downloads
wget https://github.com/cyberbotics/webots/releases/download/R2025a/webots_2025a_amd64.deb
sudo apt install ./webots_2025a_amd64.deb -y
```

### Step 3: Install ROS-Webots Bridge

```bash
sudo apt update
sudo apt install ros-humble-webots-ros2-driver
```

### Step 4: Install Python Dependencies

```bash
cd ~/ros2_ws
sudo apt install libportaudio2
pip install "numpy<2.0" torch torchaudio scikit-learn sentence-transformers scipy sounddevice requests openai python-dotenv
pip install -r requirements.txt
pip install -U nvidia-riva-client
```

### Step 5: Clone Nvidia Riva Clients (for STT/TTS)

```bash
cd ~/ros2_ws
git clone https://github.com/nvidia-riva/python-clients.git
cd python-clients && mkdir COLCON_IGNORE  # Prevent colcon from building it
```

### Step 6: Generate Action Embeddings

```bash
cd ~/ros2_ws/src/my_nao_controller/my_nao_controller
python generate_action_embeddings.py
```

---

## Building the Project

```bash
cd ~/ros2_ws

# Clean previous builds (recommended on fresh setup)
rm -rf build/ install/ log/

# Build
colcon build --symlink-install

# Source the environment
source install/setup.bash
```

### Environment Variables

```bash
echo 'export WEBOTS_HOME=/usr/local/webots' >> ~/.bashrc
source ~/.bashrc
```

---

## Usage

### Terminal 1: Launch the Robot

Starts Webots, loads the Nao robot, and attaches the animation driver.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch my_nao_controller robot_launch.py
```

Wait until you see: **"Nao Driver Ready (Action Vocabulary with Descriptions)"**

### Terminal 2: Run the AI Brain

Full conversation loop with voice input/output and gestures.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run my_nao_controller nao_brain
```

### Terminal 2 (Alternative): Test Actions Manually

Sends predefined action commands to verify robot movement.

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run my_nao_controller test_actions
```

---

## Available Actions

Defined in `nao_action_vocab.py`:

| Action | Description |
|--------|-------------|
| `stand_neutral` | Reset to standard standing posture |
| `shake_head` | Head rotates left/right (disagreement) |
| `nod_head` | Head moves up/down (agreement) |
| `wave_right_hand` | Friendly wave greeting |
| `get_out_left_hand` | Aggressive dismissal gesture (left) |
| `get_out_right_hand` | Aggressive dismissal gesture (right) |
| `explain_open_arms` | Arms spread wide (explaining/welcoming) |
| `hands_on_hips` | Power pose (authoritative) |

### Action Definition Format

```python
"action_name": {
    "description": "Text for semantic embedding search",
    "duration": 4.0,       # Total animation time (seconds)
    "repetitions": 1,      # Number of sine wave cycles
    "curves": [
        {
            "joint": "LShoulderPitch",
            "min": 0.0,
            "max": 1.5,
            "start_from_max": True  # Optional: start at peak
        }
    ]
}
```

---

## Personas

Defined in `personas.py`. Switch via `CURRENT_PERSONA_ID` in `config.py`:

| ID | Name | Personality |
|----|------|-------------|
| `polite_teacher` | Professor Nao | Patient, encouraging, educational |
| `polite_receptionist` | Receptionist Nao | Professional, helpful, courteous |
| `angry_cab_driver` | Cabbie Nao | Irritable, impatient, blunt |

Each persona includes:
- `personality`: Character traits
- `speaking_style`: How it talks
- `example_phrases`: Sample expressions
- `tts_settings`: Voice configuration (language, voice name)

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Class NaoDriver cannot be found` | Typo in class name or wrong file location | Ensure class is `class NaoDriver(WebotsController):` in `my_nao_controller/nao_driver.py` |
| `No executable found` (ros2 run) | Entry points not registered or not rebuilt | Update `setup.py`, then `rm -rf build/ install/` and `colcon build --symlink-install` |
| Webots crash / version mismatch | Incompatible Webots and ROS driver versions | Install Webots R2025a via `.deb` and `ros-humble-webots-ros2-driver` |
| `NotInitializedException` | Creating ROS node before `rclpy.init()` | Add `if not rclpy.ok(): rclpy.init(args=None)` at start of `init()` |
| `action_embeddings.pkl not found` | Embeddings not generated | Run `python generate_action_embeddings.py` in `my_nao_controller/` |
| STT/TTS timeout | API issues or network | Check `NVIDIA_API_KEY` and function IDs in `config.py` |

---

## Quick Checklist

- [ ] Ubuntu 22.04 + ROS 2 Humble installed
- [ ] Webots R2025a installed (not from apt)
- [ ] `ros-humble-webots-ros2-driver` installed
- [ ] `python-clients` cloned with `COLCON_IGNORE` marker
- [ ] `action_embeddings.pkl` generated
- [ ] `config.py` has valid Nvidia API credentials
- [ ] Workspace built with `colcon build --symlink-install`
- [ ] `WEBOTS_HOME=/usr/local/webots` exported

---

## License

MIT License

## Maintainer

Vashu Chauhan — vashu22606@iiitd.ac.in
