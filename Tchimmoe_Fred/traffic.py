import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter


IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 10  
USE_TRANSFER_LEARNING = False  

def load_data(data_dir):
    """
    Load images from dataset directory and return images with labels.
    """
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  
                    images.append(img)
                    labels.append(int(label))
    
    images = np.array(images) / 255.0  
    labels = np.array(labels)

    print(f"Dataset Size: {images.shape[0]} images")
    print(f"Class Distribution: {Counter(labels)}") 
    
    return images, labels

def create_model():
    """
    Build a CNN model with an extra Conv2D layer and reduced dropout.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),  
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),  
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_transfer_learning_model():
    """
    Create a MobileNetV2-based model for better accuracy on small datasets.
    """
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False 

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
   
    dataset_dir = r" C:\Users\utilisateur\Desktop\Traffic\Tchimmoe_Fred_Emmanuel\Tchimmoe_Fred\images"
    print("Loading data...")
    X, y = load_data(dataset_dir)
    X = X / 255.0 

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 

    
    if USE_TRANSFER_LEARNING:
        model = create_transfer_learning_model()
    else:
        model = create_model()

    
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

    
    print("Training model...")
    model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[checkpoint])
    print("Training complete. Best model saved as best_model.h5.")

if __name__ == '__main__':
    main()
