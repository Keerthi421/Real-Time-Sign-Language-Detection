#!/usr/bin/env python3
"""
Setup script for Sign Language Interpreter
This script helps set up the environment and install dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """
    Install required packages
    """
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def create_directories():
    """
    Create necessary directories
    """
    directories = ['Code', 'gestures', 'img']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def check_camera():
    """
    Check if camera is available
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera is available")
            cap.release()
            return True
        else:
            print("✗ Camera is not available")
            return False
    except ImportError:
        print("✗ OpenCV not installed")
        return False

def main():
    """
    Main setup function
    """
    print("Sign Language Interpreter Setup")
    print("=" * 40)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        print("Setup failed! Please install requirements manually.")
        return False
    
    # Check camera
    print("\n3. Checking camera...")
    if not check_camera():
        print("Warning: Camera not available. You may need to connect a camera.")
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python set_hand_histogram.py' to set up hand detection")
    print("2. Run 'python create_gestures.py' to capture training gestures")
    print("3. Run 'python rotate_images.py' to augment your dataset")
    print("4. Run 'python load_images.py' to prepare training data")
    print("5. Run 'python cnn_model_train.py' to train the model")
    print("6. Run 'python final.py' to start the interpreter")
    
    return True

if __name__ == "__main__":
    main()
