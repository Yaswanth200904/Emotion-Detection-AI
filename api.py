
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import Prediction


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model_v2.keras"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

IMG_SIZE = 48

model = None

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# --------------------------------------------------
# Load model and initialize database
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV face detector.")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Emotion detection model loaded successfully.")

    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")

    yield

    model = None


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Emotion Detection API",
    description="Detect facial emotions and store predictions in MySQL.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5502",
        "http://localhost:5502",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database": "configured",
    }


# --------------------------------------------------
# Predict emotion
# --------------------------------------------------

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Emotion detection model is not loaded.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    # Decode the uploaded image.
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image.",
        )

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces.
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected. Please position your face clearly.",
        )

    # Select the largest detected face.
    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3],
    )

    face = gray[y:y + height, x:x + width]

    # Prepare the face for the trained model.
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    # Run emotion prediction.
    try:
        prediction = model.predict(face, verbose=0)
        scores = np.asarray(prediction[0]).flatten()

        if len(scores) != len(EMOTIONS):
            raise ValueError(
                f"Expected {len(EMOTIONS)} output classes, "
                f"but the model returned {len(scores)}."
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError("The model returned invalid prediction values.")

        predicted_index = int(np.argmax(scores))
        emotion = EMOTIONS[predicted_index]
        confidence = float(scores[predicted_index])

        # Convert probabilities to regular Python floats
        # so they can be stored in the MySQL JSON column.
        probabilities = {
            EMOTIONS[index]: float(scores[index])
            for index in range(len(EMOTIONS))
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process the emotion prediction: {exc}",
        ) from exc

    # Save the prediction using the actual database column names.
    db: Session = SessionLocal()

    try:
        db_prediction = Prediction(
            emotion=emotion,
            score=confidence,
            probabilities=probabilities,
        )

        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return {
            "id": db_prediction.id,
            "emotion": emotion,
            "confidence": confidence,
            "score": confidence,
            "probabilities": probabilities,
            "created_at": (
                db_prediction.created_at.isoformat()
                if db_prediction.created_at
                else None
            ),
        }

    except Exception as exc:
        db.rollback()
        print(f"DATABASE SAVE ERROR: {exc!r}")

        raise HTTPException(
            status_code=500,
            detail="Prediction was generated, but saving it to the database failed.",
        ) from exc

    finally:
        db.close()


# --------------------------------------------------
# Get prediction history
# --------------------------------------------------

@app.get("/history")
def get_prediction_history(
    limit: int = Query(default=10, ge=1, le=100),
):
    db: Session = SessionLocal()

    try:
        records = (
            db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": record.id,
                "emotion": record.emotion,

                # The database column is score, not confidence.
                # "confidence" is retained in the API response
                # for compatibility with the frontend.
                "confidence": float(record.score),
                "score": float(record.score),

                "probabilities": record.probabilities,
                "created_at": (
                    record.created_at.isoformat()
                    if record.created_at
                    else None
                ),
            }
            for record in records
        ]

    except Exception as exc:
        print(f"HISTORY ERROR: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve prediction history.",
        ) from exc

    finally:
        db.close()
