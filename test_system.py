#!/usr/bin/env python3
"""
Test script for Sign Language Interpreter
This script tests all components of the system
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

def test_imports():
    """
    Test if all required packages can be imported
    """
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV import failed")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ NumPy import failed")
        return False
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
    except ImportError:
        print("✗ TensorFlow import failed")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        print("✓ Scikit-learn imported successfully")
    except ImportError:
        print("✗ Scikit-learn import failed")
        return False
    
    return True

def test_camera():
    """
    Test if camera is available
    """
    print("\nTesting camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera is working and can capture frames")
                cap.release()
                return True
            else:
                print("✗ Camera opened but cannot capture frames")
                cap.release()
                return False
        else:
            print("✗ Camera cannot be opened")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_directories():
    """
    Test if required directories exist
    """
    print("\nTesting directories...")
    
    required_dirs = ['Code', 'gestures', 'img']
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory '{directory}' exists")
        else:
            print(f"✗ Directory '{directory}' does not exist")
            all_exist = False
    
    return all_exist

def test_gesture_files():
    """
    Test if gesture files exist
    """
    print("\nTesting gesture files...")
    
    if not os.path.exists('gestures'):
        print("✗ Gestures directory not found")
        return False
    
    gesture_dirs = [d for d in os.listdir('gestures') if os.path.isdir(os.path.join('gestures', d))]
    
    if not gesture_dirs:
        print("✗ No gesture directories found")
        return False
    
    print(f"✓ Found {len(gesture_dirs)} gesture directories")
    
    # Check if directories have images
    total_images = 0
    for gesture_dir in gesture_dirs:
        gesture_path = os.path.join('gestures', gesture_dir)
        images = [f for f in os.listdir(gesture_path) if f.endswith('.jpg')]
        total_images += len(images)
        print(f"  - {gesture_dir}: {len(images)} images")
    
    print(f"✓ Total images: {total_images}")
    return total_images > 0

def test_model_files():
    """
    Test if model files exist
    """
    print("\nTesting model files...")
    
    model_file = 'cnn_model_keras2.h5'
    if os.path.exists(model_file):
        print(f"✓ Model file '{model_file}' exists")
        
        # Try to load the model
        try:
            model = tf.keras.models.load_model(model_file)
            print("✓ Model can be loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Model cannot be loaded: {e}")
            return False
    else:
        print(f"✗ Model file '{model_file}' not found")
        return False

def test_histogram_file():
    """
    Test if hand histogram file exists
    """
    print("\nTesting hand histogram...")
    
    histogram_file = 'hand_histogram.npy'
    if os.path.exists(histogram_file):
        print(f"✓ Hand histogram file '{histogram_file}' exists")
        return True
    else:
        print(f"✗ Hand histogram file '{histogram_file}' not found")
        return False

def test_data_files():
    """
    Test if data files exist
    """
    print("\nTesting data files...")
    
    data_files = ['Code/train_images', 'Code/train_labels', 
                  'Code/val_images', 'Code/val_labels',
                  'Code/test_images', 'Code/test_labels']
    
    all_exist = True
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✓ Data file '{data_file}' exists")
        else:
            print(f"✗ Data file '{data_file}' not found")
            all_exist = False
    
    return all_exist

def run_comprehensive_test():
    """
    Run comprehensive system test
    """
    print("Sign Language Interpreter - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Camera Test", test_camera),
        ("Directory Test", test_directories),
        ("Gesture Files Test", test_gesture_files),
        ("Model Files Test", test_model_files),
        ("Histogram Test", test_histogram_file),
        ("Data Files Test", test_data_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()
