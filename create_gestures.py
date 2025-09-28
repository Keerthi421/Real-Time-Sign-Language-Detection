import cv2
import numpy as np
import os
import pickle
from set_hand_histogram import load_hand_histogram

def create_gestures():
    """
    Create and capture gestures for training
    This function allows you to capture different ASL gestures using webcam
    """
    cap = cv2.VideoCapture(0)
    
    # Load hand histogram
    hand_hist = load_hand_histogram()
    if hand_hist is None:
        print("Please run set_hand_histogram.py first to create hand histogram!")
        return
    
    # Create gestures directory if it doesn't exist
    if not os.path.exists('gestures'):
        os.makedirs('gestures')
    
    # ASL alphabet (excluding J and Z which require movement)
    asl_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    print("Gesture Creation System")
    print("Available letters:", ', '.join(asl_letters))
    print("Instructions:")
    print("1. Enter a letter to start capturing gestures for that letter")
    print("2. Press 'c' to capture current frame")
    print("3. Press 'n' to move to next letter")
    print("4. Press 'q' to quit")
    
    current_letter = None
    gesture_count = 0
    max_gestures_per_letter = 100
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Create a copy for drawing
        frame_copy = frame.copy()
        
        # Get hand region using histogram backprojection
        hand_region = get_hand_region(frame, hand_hist)
        
        # Add text information
        if current_letter:
            cv2.putText(frame_copy, f"Capturing: {current_letter} (Count: {gesture_count})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_copy, "Press 'c' to capture, 'n' for next letter", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_copy, "Enter a letter to start capturing", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame_copy, "Press 'q' to quit", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show hand region in a separate window
        if hand_region is not None:
            cv2.imshow('Hand Region', hand_region)
        
        # Show the main frame
        cv2.imshow('Gesture Capture', frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key >= ord('a') and key <= ord('z'):
            # Set current letter
            current_letter = chr(key).upper()
            if current_letter in asl_letters:
                gesture_count = 0
                # Create directory for this letter
                letter_dir = os.path.join('gestures', str(ord(current_letter) - ord('A') + 1))
                if not os.path.exists(letter_dir):
                    os.makedirs(letter_dir)
                print(f"Started capturing gestures for letter: {current_letter}")
            else:
                print(f"Letter {current_letter} not supported. Use letters A-Y (excluding J and Z)")
                current_letter = None
                
        elif key == ord('c') and current_letter and hand_region is not None:
            # Capture gesture
            if gesture_count < max_gestures_per_letter:
                # Resize hand region to standard size
                hand_resized = cv2.resize(hand_region, (100, 100))
                
                # Convert to grayscale
                hand_gray = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2GRAY)
                
                # Save the gesture
                letter_dir = os.path.join('gestures', str(ord(current_letter) - ord('A') + 1))
                gesture_path = os.path.join(letter_dir, f"{gesture_count + 1}.jpg")
                cv2.imwrite(gesture_path, hand_gray)
                
                gesture_count += 1
                print(f"Captured gesture {gesture_count} for {current_letter}")
            else:
                print(f"Maximum gestures reached for {current_letter}")
                
        elif key == ord('n') and current_letter:
            # Move to next letter
            print(f"Completed capturing {gesture_count} gestures for {current_letter}")
            current_letter = None
            gesture_count = 0
    
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture capture completed!")

def get_hand_region(frame, hand_hist):
    """
    Extract hand region using histogram backprojection
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply histogram backprojection
    dst = cv2.calcBackProject([hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)
    
    # Apply morphological operations to clean up
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)
    
    # Threshold the backprojection
    ret, thresh = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Extract hand region
        hand_region = frame[y:y+h, x:x+w]
        
        # Draw rectangle around hand
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return hand_region
    
    return None

if __name__ == "__main__":
    create_gestures()
