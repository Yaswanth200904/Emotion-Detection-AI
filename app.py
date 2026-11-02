
import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import deque

# ---------------------------------
# 1. Find and load the trained model
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model_v2.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

# Keep the same order used during model training
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ---------------------------------
# 2. Load the face detector
# ---------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Could not load face detector")

# ---------------------------------
# 3. Open webcam
# ---------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

# Store predictions from recent frames
prediction_history = deque(maxlen=10)

try:
    while True:

        # Capture frame
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) > 0:

            # For now, process the largest detected face
            x, y, w, h = max(
                faces,
                key=lambda box: box[2] * box[3]
            )

            # Crop face
            face = gray[y:y + h, x:x + w]

            # Resize to model input dimensions
            face = cv2.resize(face, (48, 48))

            # Keep pixels in the 0-255 range.
            # The CNN's Rescaling layer handles normalization.
            face = face.astype(np.float32)

            # Shape: (1, 48, 48, 1)
            model_input = face[np.newaxis, :, :, np.newaxis]

            # Predict probabilities
            probabilities = model(
                model_input,
                training=False
            ).numpy()[0]

            # Save recent predictions
            prediction_history.append(probabilities)

            # Average the recent predictions
            smoothed_probabilities = np.mean(
                prediction_history,
                axis=0
            )

            # Find the highest-scoring expression
            predicted_index = int(
                np.argmax(smoothed_probabilities)
            )

            emotion = EMOTIONS[predicted_index]
            score = float(
                smoothed_probabilities[predicted_index]
            )

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display the predicted expression
            label = f"{emotion.title()} | Score: {score:.0%}"

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:
            # Avoid carrying old predictions into a new detection
            prediction_history.clear()

            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("AI Emotion Detector", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
