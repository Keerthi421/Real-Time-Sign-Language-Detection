import cv2
import numpy as np
import pickle
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_images():
    """
    Load and preprocess gesture images for training
    """
    gestures_path = 'gestures'
    
    if not os.path.exists(gestures_path):
        print("Gestures directory not found. Run create_gestures.py first!")
        return None, None, None, None
    
    # Get all gesture directories
    gesture_dirs = sorted(glob(os.path.join(gestures_path, '*')), key=lambda x: int(os.path.basename(x)))
    
    if not gesture_dirs:
        print("No gesture directories found!")
        return None, None, None, None
    
    print(f"Found {len(gesture_dirs)} gesture directories")
    
    images = []
    labels = []
    
    # Load images from each directory
    for i, gesture_dir in enumerate(gesture_dirs):
        if os.path.isdir(gesture_dir):
            print(f"Loading images from {gesture_dir}...")
            
            # Get all images in the directory
            image_files = glob(os.path.join(gesture_dir, '*.jpg'))
            
            for image_file in image_files:
                # Load image
                img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize image to standard size
                    img_resized = cv2.resize(img, (100, 100))
                    
                    # Normalize pixel values
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    
                    images.append(img_normalized)
                    labels.append(i)  # Use directory index as label
    
    if not images:
        print("No images found!")
        return None, None, None, None
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images with {len(gesture_dirs)} classes")
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Save the datasets
    save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    
    return (X_train, X_val, X_test, y_train, y_val, y_test)

def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save datasets to pickle files
    """
    # Create Code directory if it doesn't exist
    if not os.path.exists('Code'):
        os.makedirs('Code')
    
    # Save training data
    with open('Code/train_images', 'wb') as f:
        pickle.dump(X_train, f)
    with open('Code/train_labels', 'wb') as f:
        pickle.dump(y_train, f)
    
    # Save validation data
    with open('Code/val_images', 'wb') as f:
        pickle.dump(X_val, f)
    with open('Code/val_labels', 'wb') as f:
        pickle.dump(y_val, f)
    
    # Save test data
    with open('Code/test_images', 'wb') as f:
        pickle.dump(X_test, f)
    with open('Code/test_labels', 'wb') as f:
        pickle.dump(y_test, f)
    
    print("Datasets saved to Code/ directory")

def load_datasets():
    """
    Load pre-saved datasets
    """
    try:
        with open('Code/train_images', 'rb') as f:
            X_train = pickle.load(f)
        with open('Code/train_labels', 'rb') as f:
            y_train = pickle.load(f)
        
        with open('Code/val_images', 'rb') as f:
            X_val = pickle.load(f)
        with open('Code/val_labels', 'rb') as f:
            y_val = pickle.load(f)
        
        with open('Code/test_images', 'rb') as f:
            X_test = pickle.load(f)
        with open('Code/test_labels', 'rb') as f:
            y_test = pickle.load(f)
        
        print("Datasets loaded successfully!")
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except FileNotFoundError:
        print("No saved datasets found. Run load_images.py first!")
        return None, None, None, None, None, None

def display_gestures():
    """
    Display sample gestures from each class
    """
    gestures_path = 'gestures'
    
    if not os.path.exists(gestures_path):
        print("Gestures directory not found!")
        return
    
    # Get all gesture directories
    gesture_dirs = sorted(glob(os.path.join(gestures_path, '*')), key=lambda x: int(os.path.basename(x)))
    
    if not gesture_dirs:
        print("No gesture directories found!")
        return
    
    print(f"Displaying sample gestures from {len(gesture_dirs)} classes")
    
    # ASL alphabet mapping
    asl_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    for i, gesture_dir in enumerate(gesture_dirs):
        if os.path.isdir(gesture_dir):
            # Get sample images from this directory
            image_files = glob(os.path.join(gesture_dir, '*.jpg'))[:5]  # Show first 5 images
            
            if image_files:
                print(f"\nClass {i}: {asl_letters[i] if i < len(asl_letters) else 'Unknown'}")
                
                for j, image_file in enumerate(image_files):
                    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize for display
                        img_display = cv2.resize(img, (100, 100))
                        cv2.imshow(f'Class {i} - Sample {j+1}: {os.path.basename(image_file)}', img_display)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()

def get_image_size():
    """
    Get the standard image size used in the project
    """
    return (100, 100)

def get_num_of_classes():
    """
    Get the number of gesture classes
    """
    gestures_path = 'gestures'
    if os.path.exists(gestures_path):
        gesture_dirs = glob(os.path.join(gestures_path, '*'))
        return len([d for d in gesture_dirs if os.path.isdir(d)])
    return 0

if __name__ == "__main__":
    # Load and preprocess images
    datasets = load_images()
    
    if datasets[0] is not None:
        # Display sample gestures
        display_gestures()
        
        # Print dataset statistics
        X_train, X_val, X_test, y_train, y_val, y_test = datasets
        
        print(f"\nDataset Statistics:")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        print(f"Image size: {get_image_size()}")
