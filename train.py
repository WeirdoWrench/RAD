import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# --- 1. Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'dataset'

def load_data():
    """Loads images directly using the Kaggle dataset's Excel file."""
    images = []
    labels = []
    
    excel_path = 'dataset_database.xlsx'
    img_folder = 'dataset'
    
    print("[INFO] Reading Excel database...")
    df = pd.read_excel(excel_path)
    
    print("[INFO] Loading images into memory. Watch your RAM usage...")
    for index, row in df.iterrows():
        img_name = row['subject']    # e.g., '00001.jpg'
        label_str = row['collision'] # 'y' for collision, 'n' for normal
        
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            # Map 'y' -> 1 (accident), 'n' -> 0 (normal)
            labels.append(1 if label_str == 'y' else 0)
            
    return np.array(images), np.array(labels)

print("[INFO] Loading images...")
X, y = load_data() # We no longer need to pass DATA_DIR

# Normalize pixel values
X = X.astype('float32') / 255.0

# --- 2. Apply SMOTE ---
print("[INFO] Applying SMOTE to balance classes...")
# SMOTE requires 2D arrays, so we flatten the images, apply SMOTE, then reshape
n_samples, h, w, c = X.shape
X_flat = X.reshape((n_samples, h * w * c))

smote = SMOTE(random_state=42)
X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)

# Reshape back to image dimensions
X_resampled = X_resampled_flat.reshape((-1, h, w, c))
print(f"[INFO] New dataset shape after SMOTE: {X_resampled.shape}")

# --- 3. Train/Test Split & Data Augmentation ---
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ImageDataGenerator for training augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- 4. Build MobileNetV2 Architecture ---
print("[INFO] Compiling MobileNetV2 Model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
head = base_model.output
head = GlobalAveragePooling2D()(head)
head = Dense(128, activation='relu')(head)
head = Dropout(0.5)(head)
predictions = Dense(1, activation='sigmoid')(head)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with metrics matching your resume
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision')])

# --- 5. Training ---
# Save the best model automatically
checkpoint = ModelCheckpoint('models/accident_mobilenetv2.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("[INFO] Starting training...")
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print("[INFO] Training complete. Model saved to models/accident_mobilenetv2.h5")