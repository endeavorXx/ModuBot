# ModuBot: Modular LLM-Driven Multi-Persona Robot Framework

# Demo Videos available here - [https://drive.google.com/drive/folders/1twl8kwDbZtGfJ08aRuGSaHqhQLsAZuMV?usp=sharing](Videos)

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Humble-blue" alt="ROS2 Humble"/>
  <img src="https://img.shields.io/badge/Simulator-Webots-green" alt="Webots"/>
  <img src="https://img.shields.io/badge/Python-3.10+-yellow" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License"/>
</p>

A **modular, extensible ROS 2 framework** for controlling humanoid robots with **LLM-powered conversation**, **persona-aware behavior**, and **synchronized gesture-speech actions**. Demonstrated using the **Webots** simulator with two humanoid robots: **NAO** and **Robotis OP2**.

---

## 🎯 Key Features

- **LLM-Powered Conversations** — Natural dialogue driven by large language models with persona-specific behavior
- **Multi-Persona Support** — Easily switch between personalities (Angry Cab Driver, Polite Teacher, Polite Receptionist, etc.)
- **Semantic Action Matching** — Context-appropriate gestures selected via embedding-based intent matching
- **Gesture-Speech Synchronization** — Dynamic voice rate adjustment to sync actions with speech duration
- **Anticipatory Action Module** — Learns from failure to adapt action selection for short dialogues
- **Highly Modular Design** — Identical project structure across robots; only joint names and tuning differ

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ROBOT BRAIN                                    │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────────┐              │
│  │   VAD   │───▶│   STT   │───▶│   LLM   │───▶│ ActionSelect │              │
│  │ (Silero)│    │ (Riva)  │    │ (Llama) │    │ (Embeddings) │              │
│  └─────────┘    └─────────┘    └─────────┘    └──────┬───────┘              │
│       ▲                              │               │                      │
│       │                              ▼               ▼                      │
│   [Microphone]                 ┌─────────┐    ┌─────────────--┐             │
│                                │   TTS   │    │  Publish to   │             │
│                                │ (Riva)  │    │/perform_action│             │
│                                └────┬────┘    └──────┬──────---             │
│                                     │                │                      │
│                                     ▼                │                      │
│                               [Speaker]              │                      │
└──────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                           ROS 2 Topic: /perform_action│
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ROBOT DRIVER                                       │
│                                                                                 │
│  ┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ Action Callback│───▶│ Animation Engine│───▶│  Webots Motors  │              │
│  │ (ROS Subscriber)│   │ (Sine Wave Math)│    │ (Joint Control) │              │
│  └────────────────┘    └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Supported Robots

We demonstrate the framework with two humanoid robots. The modular design means **the project structure is identical** — only robot names, joint configurations, and action tuning differ.

| Robot | Description | Documentation |
|-------|-------------|---------------|
| **SoftBank NAO** | Popular humanoid research robot | [📖 NAO Controller README](./my_nao_controller/docs/README.md) |
| **Robotis OP2** | Open-source humanoid platform | [📖 OP2 Controller README](./op2_controller/docs/README.md) |

---

## 📊 Evaluation Metrics

Comprehensive evaluation conducted over **10 runs**, each with **20 conversation messages**, testing **3 personas** (Angry Cab Driver, Polite Teacher, Polite Receptionist).

| Metric | Score | Description |
|--------|-------|-------------|
| **Persona Fidelity** | 90% | Robot maintains persona-consistent language, tone, and behavior across interactions |
| **Action Grounding & Synchronization** | 83% ± 2% | Correct action retrieval + well-timed gesture–speech synchronization. Voice rate dynamically adjusted based on words-to-speak vs action duration |
| **Emotional TTS Quality** | 0.87 ± 0.03 | High emotional expressiveness (Whisper Large + Magpie) |
| **Interaction Latency** | 3.2 ± 0.3 sec | End-to-end STT → LLM → [TTS + Action] pipeline latency (15-20 words). Includes cloud API overhead |
| **HRI User Study** | 4.1 ± 0.5 / 5 | MOS-equivalent rating from 20 participants for likeability, clarity, and perceived intelligence |
| **Robustness & Reliability** | 81% ± 5% | Recovery from uncertainty/noise; avoidance of unsafe motions. Includes anticipatory module that learns from failure to adapt action selection for short dialogues |

---

## 🔧 Modularity & Extensibility

The framework is designed for rapid adaptation and extension:

| Task | Effort | Details |
|------|--------|---------|
| **Add New Persona** | ~20 ± 5 minutes | JSON-only configuration |
| **Add New Action** | ~30 ± 3 LOC | Action-specific driver control implementation |
| **Add New Robot** | 150–200 LOC (~30-40 min) | Previously required 8-10 hours of extensive effort per robot |

---

## 🚀 Why ROS 2?

- **Wide Community Support** — Extensive documentation, tutorials, and active development
- **Robot Agnostic** — Framework adapts to any humanoid robot with ROS 2 support
- **Modular by Design** — Nodes communicate via topics, enabling flexible system composition
- **Simulation Ready** — Seamless integration with Webots and other simulators

---

## 📁 Directory Structure 

```
.
├── ReadMe.md                    # This file
├── my_nao_controller/           # NAO Robot Controller Package
│   ├── config/
│   │   └── config.py            # API keys, paths, parameters
│   ├── docs/
│   │   └── README.md            # Detailed NAO documentation
│   ├── launch/
│   │   └── robot_launch.py      # ROS 2 launch file
│   ├── my_nao_controller/
│   │   ├── nao_brain.py         # AI node (STT → LLM → TTS → Action)
│   │   ├── nao_driver.py        # Animation engine (Webots controller)
│   │   ├── nao_action_vocab.py  # Action vocabulary definitions
│   │   ├── personas.py          # Persona definitions
│   │   ├── generate_action_embeddings.py
│   │   └── action_embeddings.pkl
│   ├── resource/
│   │   └── nao.urdf
│   ├── worlds/
│   │   └── nao_world.wbt
│   ├── package.xml
│   ├── setup.py
│   └── run.py
│
└── op2_controller/              # Robotis OP2 Controller Package
    ├── config/
    │   └── config.py
    ├── docs/
    │   └── README.md            # Detailed OP2 documentation
    ├── launch/
    │   └── robot_launch.py
    ├── op2_controller/
    │   ├── op2_brain.py
    │   ├── op2_driver.py
    │   ├── op2_action_vocab.py
    │   ├── personas.py
    │   ├── generate_action_embeddings.py
    │   └── action_embeddings.pkl
    ├── resource/
    │   └── op2.urdf
    ├── worlds/
    │   └── op2_world.wbt
    ├── package.xml
    ├── setup.py
    └── run.py
```

---

## 🛠️ System Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS (Jammy Jellyfish) |
| ROS 2 | Humble Hawksbill |
| Simulator | Webots R2023b or newer |
| Python | 3.10+ |

---

## 📚 Getting Started

1. **Clone the repository** into your ROS 2 workspace:
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/endeavorXx/ROS-Nao-Simulation-in-Webots.git
   ```

2. **Follow robot-specific instructions**:
   - [NAO Controller Setup](./my_nao_controller/docs/README.md#installation)
   - [OP2 Controller Setup](./op2_controller/docs/README.md#installation)

3. **Build the workspace**:
   ```bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

---

## 📖 Documentation

For detailed setup, configuration, and usage instructions, refer to the robot-specific documentation:

- **[NAO Robot Controller Documentation](./my_nao_controller/docs/README.md)**
- **[Robotis OP2 Controller Documentation](./op2_controller/docs/README.md)**

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Built with ROS 2 Humble, Webots, and various open-source AI/ML libraries including Silero VAD, NVIDIA Riva, and Sentence Transformers.
