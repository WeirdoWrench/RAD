import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'models/accident_mobilenetv2.h5'

# IMPORTANT: Change this path to any image on your computer you want to test!
# You can pick one from your dataset folder to verify it works.
IMAGE_PATH = 'dataset/00001.jpg' 
IMG_SIZE = 224

print(f"[INFO] Loading model from {MODEL_PATH}...")
# Load the model you just trained
model = tf.keras.models.load_model(MODEL_PATH)

print(f"[INFO] Loading image from {IMAGE_PATH}...")
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("\n[ERROR] Could not read the image. Double-check the IMAGE_PATH!")
else:
    # --- Preprocessing (Must match training exactly) ---
    # OpenCV loads images in BGR format, but we trained on RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    resized_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    normalized_img = resized_img.astype('float32') / 255.0
    
    # Add batch dimension: changes shape from (224, 224, 3) to (1, 224, 224, 3)
    input_tensor = np.expand_dims(normalized_img, axis=0)

    # --- Inference ---
    print("[INFO] Running prediction...")
    prediction = model.predict(input_tensor, verbose=0)[0][0]

    # --- Results ---
    print("\n" + "="*40)
    if prediction > 0.5:
        print(f"🚨 RESULT: ACCIDENT DETECTED!")
        print(f"Confidence:  {prediction * 100:.2f}%")
    else:
        print(f"✅ RESULT: NORMAL TRAFFIC")
        print(f"Confidence:  {(1 - prediction) * 100:.2f}%")
    print("="*40 + "\n")
        
    # Show the image on screen until you press any key
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()