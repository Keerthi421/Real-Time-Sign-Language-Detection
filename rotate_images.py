import cv2
import numpy as np
import os
from glob import glob

def augment_images():
    """
    Augment gesture images by rotating and flipping them
    This increases the dataset size and improves model robustness
    """
    gestures_path = 'gestures'
    
    if not os.path.exists(gestures_path):
        print("Gestures directory not found. Run create_gestures.py first!")
        return
    
    # Get all gesture directories
    gesture_dirs = glob(os.path.join(gestures_path, '*'))
    
    print(f"Found {len(gesture_dirs)} gesture directories")
    
    for gesture_dir in gesture_dirs:
        if os.path.isdir(gesture_dir):
            print(f"Processing {gesture_dir}...")
            
            # Get all images in the directory
            image_files = glob(os.path.join(gesture_dir, '*.jpg'))
            
            for image_file in image_files:
                # Load the image
                img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Get the base filename without extension
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                dir_path = os.path.dirname(image_file)
                
                # Original image (already exists)
                original_path = image_file
                
                # 1. Horizontal flip
                flipped_h = cv2.flip(img, 1)
                flipped_h_path = os.path.join(dir_path, f"{base_name}_flip_h.jpg")
                cv2.imwrite(flipped_h_path, flipped_h)
                
                # 2. Vertical flip
                flipped_v = cv2.flip(img, 0)
                flipped_v_path = os.path.join(dir_path, f"{base_name}_flip_v.jpg")
                cv2.imwrite(flipped_v_path, flipped_v)
                
                # 3. Both flips
                flipped_both = cv2.flip(img, -1)
                flipped_both_path = os.path.join(dir_path, f"{base_name}_flip_both.jpg")
                cv2.imwrite(flipped_both_path, flipped_both)
                
                # 4. Rotate 90 degrees clockwise
                rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                rotated_90_path = os.path.join(dir_path, f"{base_name}_rot_90.jpg")
                cv2.imwrite(rotated_90_path, rotated_90)
                
                # 5. Rotate 90 degrees counter-clockwise
                rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_270_path = os.path.join(dir_path, f"{base_name}_rot_270.jpg")
                cv2.imwrite(rotated_270_path, rotated_270)
                
                # 6. Rotate 180 degrees
                rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
                rotated_180_path = os.path.join(dir_path, f"{base_name}_rot_180.jpg")
                cv2.imwrite(rotated_180_path, rotated_180)
                
                # 7. Add slight rotation variations
                for angle in [-15, -10, -5, 5, 10, 15]:
                    rotated = rotate_image(img, angle)
                    if rotated is not None:
                        rotated_path = os.path.join(dir_path, f"{base_name}_rot_{angle}.jpg")
                        cv2.imwrite(rotated_path, rotated)
                
                # 8. Add brightness variations
                for brightness in [0.8, 0.9, 1.1, 1.2]:
                    bright_img = adjust_brightness(img, brightness)
                    bright_path = os.path.join(dir_path, f"{base_name}_bright_{int(brightness*100)}.jpg")
                    cv2.imwrite(bright_path, bright_img)
                
                # 9. Add contrast variations
                for contrast in [0.8, 0.9, 1.1, 1.2]:
                    contrast_img = adjust_contrast(img, contrast)
                    contrast_path = os.path.join(dir_path, f"{base_name}_contrast_{int(contrast*100)}.jpg")
                    cv2.imwrite(contrast_path, contrast_img)
    
    print("Image augmentation completed!")

def rotate_image(image, angle):
    """
    Rotate image by given angle
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate image
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Crop to original size
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    cropped = rotated[start_y:start_y+h, start_x:start_x+w]
    
    return cropped

def adjust_brightness(image, brightness_factor):
    """
    Adjust image brightness
    """
    if image is None:
        return None
    
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Apply brightness adjustment
    bright_img = img_float * brightness_factor
    
    # Clip values to valid range
    bright_img = np.clip(bright_img, 0, 255)
    
    # Convert back to uint8
    return bright_img.astype(np.uint8)

def adjust_contrast(image, contrast_factor):
    """
    Adjust image contrast
    """
    if image is None:
        return None
    
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Apply contrast adjustment
    contrast_img = (img_float - 128) * contrast_factor + 128
    
    # Clip values to valid range
    contrast_img = np.clip(contrast_img, 0, 255)
    
    # Convert back to uint8
    return contrast_img.astype(np.uint8)

def display_augmented_images(gesture_dir, num_samples=5):
    """
    Display sample augmented images for a gesture
    """
    if not os.path.exists(gesture_dir):
        print(f"Directory {gesture_dir} not found!")
        return
    
    # Get sample images
    image_files = glob(os.path.join(gesture_dir, '*.jpg'))[:num_samples]
    
    if not image_files:
        print(f"No images found in {gesture_dir}")
        return
    
    print(f"Displaying {len(image_files)} sample images from {gesture_dir}")
    
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imshow(f'Sample {i+1}: {os.path.basename(image_file)}', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    augment_images()
    
    # Display sample augmented images
    gesture_dirs = glob('gestures/*')
    if gesture_dirs:
        print(f"\nDisplaying sample augmented images from {gesture_dirs[0]}")
        display_augmented_images(gesture_dirs[0])
