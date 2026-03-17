import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'models/accident_mobilenetv2.h5'
VIDEO_SOURCE = 'test_video.mp4' # Change to 0 to use your laptop webcam
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.30 # Adjust to reduce false positives

print("[INFO] Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

alert_buffer = 0  # <--- NEW: This will remember if an accident just happened
BUFFER_FRAMES = 50 # <--- NEW: Keep the alarm on for 50 frames after a hit

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # --- NEW: Scale down the display window for your screen ---
    h, w = frame.shape[:2]
    target_height = 800 # A good size that fits on modern screens
    scale = target_height / h
    new_width = int(w * scale)
    
    # We draw our text and boxes on this smaller frame
    display_frame = cv2.resize(frame, (new_width, target_height))
    
    # --- Preprocess the ORIGINAL frame for the model ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
    normalized_frame = resized_frame.astype('float32') / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)
    
    # Predict
    prediction = model.predict(input_tensor, verbose=0)[0][0]
    
    # --- NEW LOGIC: Trigger the buffer ---
    if prediction >= CONFIDENCE_THRESHOLD:
        alert_buffer = BUFFER_FRAMES # Reset the timer to maximum
        current_conf = prediction

    # --- NEW LOGIC: Display based on the buffer, not just the single frame ---
    if alert_buffer > 0:
        label = f"ACCIDENT DETECTED! (Alert active)"
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 60), (0, 0, 255), -1)
        cv2.putText(display_frame, "EMERGENCY: COLLISION DETECTED", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        alert_buffer -= 1 # Countdown the timer
    else:
        label = f"Normal Traffic: {(1-prediction)*100:.1f}%"
        cv2.putText(display_frame, label, (20, display_frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(display_frame, f"Raw AI Score: {prediction:.3f}", (20, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("RAD - Real-Time Accident Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()