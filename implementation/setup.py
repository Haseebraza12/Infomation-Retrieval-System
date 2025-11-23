#!/usr/bin/env python3
"""
Setup and Installation Script for Cortex IR System
Automates the installation and setup process
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"→ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} failed")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def main():
    """Main setup process"""
    print_header("CORTEX IR SYSTEM - SETUP SCRIPT")
    
    print("This script will:")
    print("  1. Check Python version")
    print("  2. Create virtual environment")
    print("  3. Install dependencies")
    print("  4. Download spaCy model")
    print("  5. Create .env file")
    print("  6. Optionally preprocess data and build indices")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Step 1: Check Python version
    print_header("Step 1: Checking Python Version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create virtual environment
    print_header("Step 2: Creating Virtual Environment")
    if not Path("venv").exists():
        if not run_command(
            f"{sys.executable} -m venv venv",
            "Creating virtual environment"
        ):
            sys.exit(1)
    else:
        print("  ✓ Virtual environment already exists")
    
    # Determine activation command
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Step 3: Upgrade pip
    print_header("Step 3: Upgrading pip")
    run_command(
        f"{pip_cmd} install --upgrade pip",
        "Upgrading pip"
    )
    
    # Step 4: Install dependencies
    print_header("Step 4: Installing Dependencies")
    if not run_command(
        f"{pip_cmd} install -r requirements.txt",
        "Installing Python packages"
    ):
        print("\n⚠ Some packages may have failed to install.")
        print("You can manually install them later.")
    
    # Step 5: Download spaCy model
    print_header("Step 5: Downloading spaCy Model")
    run_command(
        f"{python_cmd} -m spacy download en_core_web_sm",
        "Downloading en_core_web_sm"
    )
    
    # Step 6: Create .env file
    print_header("Step 6: Creating Environment File")
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("  ✓ Created .env from .env.example")
        else:
            print("  ⚠ .env.example not found, skipping")
    else:
        print("  ✓ .env file already exists")
    
    # Step 7: Optional - Preprocess and index
    print_header("Step 7: Data Preprocessing (Optional)")
    print("\nDo you want to preprocess the articles and build indices now?")
    print("(This takes ~4-6 minutes for 2000 articles)")
    choice = input("\nRun preprocessing? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n→ Running preprocessing...")
        if run_command(
            f"{python_cmd} preprocessing.py",
            "Preprocessing articles"
        ):
            print("\n→ Running indexing...")
            run_command(
                f"{python_cmd} indexing.py",
                "Building indices"
            )
    else:
        print("\n  ⚠ Skipped preprocessing. Run manually later:")
        print(f"     {python_cmd} preprocessing.py")
        print(f"     {python_cmd} indexing.py")
    
    # Final instructions
    print_header("SETUP COMPLETE!")
    
    print("✓ Installation successful!\n")
    print("Next steps:")
    print(f"  1. Activate virtual environment: {activate_cmd}")
    
    if choice != 'y':
        print(f"  2. Run preprocessing: {python_cmd} preprocessing.py")
        print(f"  3. Build indices: {python_cmd} indexing.py")
        print(f"  4. Launch web interface: {python_cmd} gradio_app.py")
    else:
        print(f"  2. Launch web interface: {python_cmd} gradio_app.py")
    
    print("\nAlternatively, test the pipeline:")
    print(f"  {python_cmd} main.py")
    
    print("\nFor help, see README.md")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Setup failed: {e}")
        sys.exit(1)
