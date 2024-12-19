import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('accidetect.keras')  # Replace with your model's filename

# Open the test video
cap = cv2.VideoCapture('test_vedio.mp4')  # Replace with your test video filename

# Check if the video was loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(gray_frame, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (grayscale)

    # Make predictions
    prediction = model.predict(img, verbose=0)
    label = "Accident" if prediction[0] > 0.5 else "No Accident"
    confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]

    # Annotate the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Test Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Inside the loop, write the frame
out.write(frame)

# Release the writer
out.release()
