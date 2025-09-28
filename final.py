import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from set_hand_histogram import load_hand_histogram
from cnn_model_train import load_trained_model, predict_gesture

class SignLanguageInterpreter:
    def __init__(self):
        """
        Initialize the Sign Language Interpreter
        """
        self.cap = None
        self.hand_hist = None
        self.model = None
        self.asl_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        self.prediction_history = []
        self.confidence_threshold = 0.7
        
    def initialize(self):
        """
        Initialize camera, hand histogram, and model
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera!")
            return False
        
        # Load hand histogram
        self.hand_hist = load_hand_histogram()
        if self.hand_hist is None:
            print("Error: Could not load hand histogram. Run set_hand_histogram.py first!")
            return False
        
        # Load trained model
        self.model = load_trained_model()
        if self.model is None:
            print("Error: Could not load trained model. Run cnn_model_train.py first!")
            return False
        
        print("Sign Language Interpreter initialized successfully!")
        return True
    
    def get_hand_region(self, frame):
        """
        Extract hand region using histogram backprojection
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply histogram backprojection
        dst = cv2.calcBackProject([hsv], [0, 1], self.hand_hist, [0, 180, 0, 256], 1)
        
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
            
            return hand_region, (x, y, w, h)
        
        return None, None
    
    def smooth_prediction(self, prediction, confidence):
        """
        Smooth predictions using history to reduce flickering
        """
        self.prediction_history.append((prediction, confidence))
        
        # Keep only last 5 predictions
        if len(self.prediction_history) > 5:
            self.prediction_history.pop(0)
        
        # If confidence is too low, don't update
        if confidence < self.confidence_threshold:
            return None, 0
        
        # Return the most common prediction in recent history
        if len(self.prediction_history) >= 3:
            recent_predictions = [p[0] for p in self.prediction_history[-3:]]
            most_common = max(set(recent_predictions), key=recent_predictions.count)
            avg_confidence = np.mean([p[1] for p in self.prediction_history[-3:] if p[0] == most_common])
            return most_common, avg_confidence
        
        return prediction, confidence
    
    def run(self):
        """
        Run the real-time sign language interpreter
        """
        if not self.initialize():
            return
        
        print("Sign Language Interpreter Started!")
        print("Instructions:")
        print("- Make ASL gestures in front of the camera")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset prediction history")
        print("- Press 't' to toggle confidence threshold")
        
        current_prediction = None
        current_confidence = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get hand region
            hand_region, hand_rect = self.get_hand_region(frame)
            
            if hand_region is not None:
                # Predict gesture
                prediction, confidence = predict_gesture(self.model, hand_region)
                
                if prediction is not None:
                    # Smooth prediction
                    smoothed_pred, smoothed_conf = self.smooth_prediction(prediction, confidence)
                    
                    if smoothed_pred is not None:
                        current_prediction = smoothed_pred
                        current_confidence = smoothed_conf
            
            # Display current prediction
            if current_prediction is not None and current_confidence > self.confidence_threshold:
                letter = self.asl_letters[current_prediction] if current_prediction < len(self.asl_letters) else 'Unknown'
                
                # Display prediction with confidence
                cv2.putText(frame, f"Prediction: {letter}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No gesture detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset, 't' to toggle threshold", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Sign Language Interpreter', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset prediction history
                self.prediction_history = []
                current_prediction = None
                current_confidence = 0
                print("Prediction history reset!")
            elif key == ord('t'):
                # Toggle confidence threshold
                self.confidence_threshold = 0.5 if self.confidence_threshold == 0.7 else 0.7
                print(f"Confidence threshold set to: {self.confidence_threshold}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Sign Language Interpreter stopped!")

def main():
    """
    Main function to run the Sign Language Interpreter
    """
    interpreter = SignLanguageInterpreter()
    interpreter.run()

if __name__ == "__main__":
    main()
