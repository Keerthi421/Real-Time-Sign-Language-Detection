import numpy as np
import pickle
import cv2
import os
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from load_images import load_datasets, get_image_size, get_num_of_classes

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def cnn_model():
    """
    Create CNN model for sign language recognition
    """
    image_x, image_y = get_image_size()
    num_of_classes = get_num_of_classes()
    
    if num_of_classes == 0:
        print("No gesture classes found. Run create_gestures.py first!")
        return None, None
    
    model = Sequential([
        # First convolutional block
        Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        # Second convolutional block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),
        
        # Third convolutional block
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_of_classes, activation='softmax')
    ])
    
    # Compile model
    sgd = optimizers.SGD(learning_rate=1e-2)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """
    Train the CNN model
    """
    # Load datasets
    datasets = load_datasets()
    
    if datasets[0] is None:
        print("No datasets found. Run load_images.py first!")
        return None
    
    X_train, X_val, X_test, y_train, y_val, y_test = datasets
    
    # Reshape data for CNN (add channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {y_train_cat.shape[1]}")
    
    # Create model
    model = cnn_model()
    
    if model is None:
        return None
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    filepath = "cnn_model_keras2.h5"
    checkpoint = ModelCheckpoint(
        filepath, 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    callbacks_list = [checkpoint, early_stopping]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=15,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save final model
    model.save('cnn_model_keras2.h5')
    print("Model saved as 'cnn_model_keras2.h5'")
    
    return model, history

def load_trained_model():
    """
    Load pre-trained model
    """
    if os.path.exists('cnn_model_keras2.h5'):
        model = keras.models.load_model('cnn_model_keras2.h5')
        print("Pre-trained model loaded successfully!")
        return model
    else:
        print("No pre-trained model found. Run cnn_model_train.py first!")
        return None

def predict_gesture(model, image):
    """
    Predict gesture from image
    """
    if model is None:
        return None, None
    
    # Preprocess image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_resized = cv2.resize(image, (100, 100))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_reshaped = image_normalized.reshape(1, 100, 100, 1)
    
    # Make prediction
    prediction = model.predict(image_reshaped, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return predicted_class, confidence

def plot_training_history(history):
    """
    Plot training history
    """
    import matplotlib.pyplot as plt
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # Train the model
    result = train_model()
    
    if result is not None:
        model, history = result
        
        # Plot training history
        try:
            plot_training_history(history)
        except ImportError:
            print("Matplotlib not available for plotting")
        
        print("Training completed successfully!")
    else:
        print("Training failed!")
