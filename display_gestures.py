import cv2
import numpy as np
import os
from glob import glob
from load_images import display_gestures

def main():
    """
    Display sample gestures from the dataset
    """
    print("Gesture Display Tool")
    print("=" * 30)
    
    # Check if gestures directory exists
    if not os.path.exists('gestures'):
        print("Gestures directory not found!")
        print("Please run create_gestures.py first to capture some gestures.")
        return
    
    # Get all gesture directories
    gesture_dirs = glob('gestures/*')
    if not gesture_dirs:
        print("No gesture directories found!")
        print("Please run create_gestures.py first to capture some gestures.")
        return
    
    print(f"Found {len(gesture_dirs)} gesture directories")
    
    # ASL alphabet mapping
    asl_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Display gestures from each class
    for i, gesture_dir in enumerate(sorted(gesture_dirs, key=lambda x: int(os.path.basename(x)))):
        if os.path.isdir(gesture_dir):
            # Get sample images from this directory
            image_files = glob(os.path.join(gesture_dir, '*.jpg'))
            
            if image_files:
                letter = asl_letters[i] if i < len(asl_letters) else 'Unknown'
                print(f"\nClass {i}: {letter} ({len(image_files)} images)")
                
                # Show first 5 images
                for j, image_file in enumerate(image_files[:5]):
                    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize for display
                        img_display = cv2.resize(img, (200, 200))
                        cv2.imshow(f'Class {i} ({letter}) - Sample {j+1}', img_display)
                
                print("Press any key to continue to next class...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    print("\nGesture display completed!")

if __name__ == "__main__":
    main()
