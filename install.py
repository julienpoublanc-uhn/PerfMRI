import os
import platform
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print(f"\n>> {cmd}")
    subprocess.check_call(cmd, shell=True)

print("======================================")
print(" PerfMRI installation")
print("======================================")

os_name = platform.system()
print(f"\nDetected OS: {os_name}")

# --------------------------------------
# macOS
# --------------------------------------
if os_name == "Darwin":
    print("\nInstalling Python 3.10 via brew...")
    run("brew update")
    run("brew install python@3.10 python-tk@3.10")

# --------------------------------------
# Linux
# --------------------------------------
elif os_name == "Linux":
    print("\nInstalling dependencies via apt...")
    run("sudo apt update")
    run("sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-tk git")

# --------------------------------------
# Windows
# --------------------------------------
elif os_name == "Windows":
    print("\nWindows detected.")
    print("Make sure Python 3.10 + git are installed and in PATH.")

# --------------------------------------
# Virtual env
# --------------------------------------
print("\nCreating virtual environment...")
if os_name == "Windows":
    run("py -3.10 -m venv perfmri_env")
else:
    run("python3.10 -m venv perfmri_env")

activate = "perfmri_env/Scripts/activate" if os_name == "Windows" else "perfmri_env/bin/activate"

print("\nActivating environment...")

if os_name == "Windows":
    pip = "perfmri_env\\Scripts\\pip"
    python = "perfmri_env\\Scripts\\python"
else:
    pip = "perfmri_env/bin/pip"
    python = "perfmri_env/bin/python"

# --------------------------------------
# Install packages
# --------------------------------------
run(f"{pip} install --upgrade pip")
run(f"{pip} install -r requirements.txt")
run(f"{pip} install git+https://github.com/nipy/nipy.git@0.6.1")

# --------------------------------------
# Sanity check
# --------------------------------------
run(f"{python} -c \"import numpy, matplotlib, tkinter, nipy; print('PerfMRI environment OK')\"")

print("\n======================================")
print(" Installation completed successfully")
print("======================================")
