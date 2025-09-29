import os
import sys
from pathlib import Path

def _check_devcontainer():
    if not all([
        Path("/root/.mainrun").exists()
    ]):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("""
🚨 DEVCONTAINER REQUIRED 🚨

This project must run in its devcontainer for:

✓ Assessment submission  ✓ Metrics collection  ✓ Review process

Setup Instructions:

📖 https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container

📋 IMPORTANT: Read README.md for Mainrun instructions and review process

☠️☠️  Running outside devcontainer = broken submission & metrics  ☠️☠️
        """)
        sys.exit(1)

_check_devcontainer()