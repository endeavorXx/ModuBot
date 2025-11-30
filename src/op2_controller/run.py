#!/usr/bin/env python3
"""
run.py - One-step helper to prepare and run the OP2 simulation project

This script automates the common setup tasks for the repository so users
don't have to install each Python dependency manually.

Features:
 - Create a Python virtual environment (`.venv`) and install Python deps
 - Optionally clone Nvidia `python-clients` repo used for STT/TTS examples
 - Generate action embeddings used for semantic action selection
 - Build the workspace with `colcon build --symlink-install`
 - Launch Webots + driver or run the brain/test script via `ros2 run`

Usage examples:
  # Create venv, install python deps, clone clients, generate embeddings, build
  ./run.py --setup --generate-embeddings --build

  # Launch Webots + driver
  ./run.py --launch

  # Run the brain (LLM + TTS) node
  ./run.py --run-brain

  # Do everything then launch the simulation
  ./run.py --all
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / '.venv'
PYTHON = sys.executable


def check_cmd(name):
    """Return True if command `name` is available on PATH."""
    return shutil.which(name) is not None


def run(cmd, check=True, env=None, shell=False):
    """Run command and stream output."""
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, check=check, env=env, shell=shell)


def create_venv():
    if VENV_DIR.exists():
        print("Virtual environment already exists at .venv")
        return
    print("Creating virtual environment at .venv...")
    try:
        run([PYTHON, '-m', 'venv', str(VENV_DIR)])
    except subprocess.CalledProcessError:
        print("\n[ERROR] Failed to create virtual environment.")
        print("You might be missing the python3-venv package.")
        print("Try installing it with:\n")
        print("    sudo apt update && sudo apt install python3-venv\n")
        sys.exit(1)
    print("Virtual environment created.")


def pip_install(packages):
    pip = VENV_DIR / 'bin' / 'pip'
    if not pip.exists():
        create_venv()
    print("Installing Python packages into virtualenv (.venv)...")
    cmd = [str(pip), 'install', '--upgrade'] + packages
    run(cmd)


def ensure_python_deps():
    create_venv()
    # Minimal set required by the project Python files
    packages = [
        'numpy<2.0',
        'scipy',
        'sounddevice',
        'sentence-transformers',
        'scikit-learn',
        'openai',
        'torch',
        'nvidia-riva-client',  # Required for STT/TTS with Nvidia Riva
    ]
    pip_install(packages)


def clone_python_clients():
    target = ROOT / 'python-clients'
    if target.exists():
        print('python-clients already present; skipping clone')
        return
    print('Cloning Nvidia python-clients (used by STT/TTS scripts)...')
    run(['git', 'clone', 'https://github.com/nvidia-riva/python-clients.git', str(target)])
    # Add COLCON_IGNORE to avoid being built by colcon
    ignore = target / 'COLCON_IGNORE'
    ignore.write_text('')
    print('Cloned and added COLCON_IGNORE.')


def generate_embeddings():
    gen_script = ROOT / 'op2_controller' / 'generate_action_embeddings.py'
    if not gen_script.exists():
        print('Embedding generation script not found:', gen_script)
        return
    
    if not VENV_DIR.exists():
        ensure_python_deps()

    print('Generating action embeddings...')
    python = VENV_DIR / 'bin' / 'python'
    run([str(python), str(gen_script)], check=True)
    # Move generated file into package if not already
    generated = Path('action_embeddings.pkl')
    if generated.exists():
        dest = ROOT / 'op2_controller' / 'action_embeddings.pkl'
        shutil.move(str(generated), str(dest))
        print(f'Moved embeddings to {dest}')


def build_colcon():
    if not check_cmd('colcon'):
        print('colcon not found on PATH. Please install colcon (apt or pip) and retry.')
        return
    print('Building workspace with colcon...')
    # Run from repository root (assumes this repo is the package root)
    cwd = ROOT
    for d in ['build', 'install', 'log']:
        (cwd / d).mkdir(exist_ok=True)
    # Clean previous builds to be safe
    run(['rm', '-rf', 'build', 'install', 'log'], check=False, shell=False)
    run(['colcon', 'build', '--symlink-install'], check=True, env=None)
    print('Build finished. To use the workspace interactively run:')
    print('  source install/setup.bash')


def launch_simulation():
    if not check_cmd('ros2'):
        print('ros2 command not found. Source your ROS 2 installation or install ROS 2.')
        return
    # Launch via a bash shell that sources the install/setup.bash
    cmd = "bash -lc 'source install/setup.bash && ros2 launch op2_controller robot_launch.py'"
    run(cmd, shell=True)


def run_brain():
    if not check_cmd('ros2'):
        print('ros2 command not found. Source your ROS 2 installation or install ROS 2.')
        return
    cmd = "bash -lc 'source install/setup.bash && ros2 run op2_controller op2_brain'"
    run(cmd, shell=True)


def run_test_actions():
    if not check_cmd('ros2'):
        print('ros2 command not found. Source your ROS 2 installation or install ROS 2.')
        return
    cmd = "bash -lc 'source install/setup.bash && ros2 run op2_controller test_actions'"
    run(cmd, shell=True)


def parse_args():
    p = argparse.ArgumentParser(description='One-step helper for op2_controller')
    p.add_argument('--setup', action='store_true', help='Create venv and install Python deps')
    p.add_argument('--clone-clients', action='store_true', help='Clone Nvidia python-clients repo')
    p.add_argument('--generate-embeddings', action='store_true', help='Run generate_action_embeddings')
    p.add_argument('--build', action='store_true', help='Build with colcon')
    p.add_argument('--launch', action='store_true', help='Launch Webots + driver')
    p.add_argument('--run-brain', action='store_true', help='Run the op2_brain node')
    p.add_argument('--test-actions', action='store_true', help='Run the test_actions script')
    p.add_argument('--all', action='store_true', help='Do setup, clone, generate, build, then launch')
    return p.parse_args()


def main():
    args = parse_args()

    if args.all:
        args.setup = args.clone_clients = args.generate_embeddings = args.build = True

    if args.setup:
        ensure_python_deps()

    if args.clone_clients:
        clone_python_clients()

    if args.generate_embeddings:
        generate_embeddings()

    if args.build:
        build_colcon()

    if args.launch:
        launch_simulation()

    if args.run_brain:
        run_brain()

    if args.test_actions:
        run_test_actions()

    if not any(vars(args).values()):
        print('No action specified. Use --help to see options.')


if __name__ == '__main__':
    main()
