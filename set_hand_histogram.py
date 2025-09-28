import cv2
import numpy as np
import os

def set_hand_histogram():
    """
    Set up hand histogram for skin color detection
    This function captures hand samples to create a histogram for skin color detection
    """
    cap = cv2.VideoCapture(0)
    
    # Create window for histogram setup
    cv2.namedWindow('Hand Histogram Setup', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Histogram Setup', 800, 600)
    
    print("Hand Histogram Setup")
    print("Instructions:")
    print("1. Place your hand in the green box")
    print("2. Press 'c' to capture hand samples")
    print("3. Press 's' to save histogram")
    print("4. Press 'q' to quit")
    
    # Variables for histogram calculation
    hand_hist = None
    hand_rect_one_x = None
    hand_rect_one_y = None
    hand_rect_two_x = None
    hand_rect_two_y = None
    
    # Rectangle coordinates for hand detection
    hand_rect_one_x = 300
    hand_rect_one_y = 100
    hand_rect_two_x = 500
    hand_rect_two_y = 300
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Create a copy for drawing
        frame_copy = frame.copy()
        
        # Draw rectangle for hand placement
        cv2.rectangle(frame_copy, (hand_rect_one_x, hand_rect_one_y), 
                     (hand_rect_two_x, hand_rect_two_y), (0, 255, 0), 2)
        
        # Add text instructions
        cv2.putText(frame_copy, "Place hand in green box and press 'c'", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_copy, "Press 's' to save histogram", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_copy, "Press 'q' to quit", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Hand Histogram Setup', frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Capture hand region for histogram
            hand_region = frame[hand_rect_one_y:hand_rect_two_y, 
                              hand_rect_one_x:hand_rect_two_x]
            
            if hand_region.size > 0:
                # Convert to HSV for better skin detection
                hsv_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
                
                # Calculate histogram
                hist = cv2.calcHist([hsv_hand], [0, 1], None, [180, 256], [0, 180, 0, 256])
                
                if hand_hist is None:
                    hand_hist = hist
                else:
                    hand_hist += hist
                
                print("Hand sample captured!")
                
        elif key == ord('s'):
            if hand_hist is not None:
                # Normalize histogram
                hand_hist = cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
                
                # Save histogram
                np.save('hand_histogram.npy', hand_hist)
                print("Histogram saved as 'hand_histogram.npy'")
                break
            else:
                print("No histogram data to save. Capture some hand samples first!")
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return hand_hist

def load_hand_histogram():
    """
    Load pre-saved hand histogram
    """
    if os.path.exists('hand_histogram.npy'):
        return np.load('hand_histogram.npy')
    else:
        print("No saved histogram found. Run set_hand_histogram() first.")
        return None

if __name__ == "__main__":
    set_hand_histogram()
