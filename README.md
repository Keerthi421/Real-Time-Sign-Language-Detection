# Sign Language Interpreter using Deep Learning

A sign language interpreter using live video feed from the camera. This project uses CNN (Convolutional Neural Networks) to recognize American Sign Language gestures in real-time.

## Features

- Real-time sign language recognition using webcam
- CNN-based deep learning model with >95% accuracy
- Support for 44 ASL characters
- Live video processing with OpenCV
- Gesture capture and training system

## Technologies Used

- Python
- TensorFlow/Keras
- OpenCV
- NumPy

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up hand histogram:
```bash
python set_hand_histogram.py
```

3. Create gestures:
```bash
python create_gestures.py
```

4. Augment images:
```bash
python rotate_images.py
```

5. Load and split data:
```bash
python load_images.py
```

6. Train the model:
```bash
python cnn_model_train.py
```

7. Run the interpreter:
```bash
python final.py
```

## Project Structure

- `Code/` - Main source code directory
- `gestures/` - Training gesture images
- `img/` - Project images and assets
- `set_hand_histogram.py` - Hand detection setup
- `create_gestures.py` - Gesture capture system
- `rotate_images.py` - Image augmentation
- `load_images.py` - Data preprocessing
- `cnn_model_train.py` - Model training
- `final.py` - Real-time recognition app

## Usage

1. Run the setup scripts in order
2. Capture gestures using the webcam interface
3. Train the CNN model on your captured data
4. Run the final application for real-time recognition

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with MaxPooling
- Dense layers with Dropout for regularization
- Softmax output for multi-class classification

## License

MIT License
