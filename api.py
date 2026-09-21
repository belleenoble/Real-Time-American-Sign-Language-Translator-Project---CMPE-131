from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
SIGN_MODEL_PATH = BASE_DIR / "models" / "static_sign_classifier.joblib"
CONFIDENCE_THRESHOLD = 0.60

app = Flask(__name__)
CORS(app)

sign_model = joblib.load(SIGN_MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(landmarker_options)


def normalize_landmarks(landmarks, hand_name):
    coordinates = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float32,
    )

    # Move the wrist to the origin.
    coordinates -= coordinates[0]

    # Mirror left hands so both orientations are comparable.
    if hand_name == "Left":
        coordinates[:, 0] *= -1

    # Remove differences caused by hand size and camera distance.
    scale = np.max(np.linalg.norm(coordinates, axis=1))

    if scale > 0:
        coordinates /= scale

    return coordinates.flatten()


@app.get("/health")
def health():
    return jsonify({
        "status": "online",
        "message": "ASL recognition API is running",
    })


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "No image was provided",
        }), 400

    uploaded_image = request.files["image"]
    image_bytes = uploaded_image.read()

    frame = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        return jsonify({
            "error": "The uploaded image could not be decoded",
        }), 400

    # Match the preprocessing used during data collection and training.
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    media_pipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = landmarker.detect(media_pipe_image)

    if not result.hand_landmarks:
        return jsonify({
            "handDetected": False,
            "letter": None,
            "confidence": 0.0,
            "recognized": False,
        })

    landmarks = result.hand_landmarks[0]
    hand_name = result.handedness[0][0].category_name
    features = normalize_landmarks(landmarks, hand_name)

    probabilities = sign_model.predict_proba([features])[0]
    best_index = int(np.argmax(probabilities))

    predicted_letter = str(sign_model.classes_[best_index])
    confidence = float(probabilities[best_index])
    recognized = confidence >= CONFIDENCE_THRESHOLD

    return jsonify({
        "handDetected": True,
        "letter": predicted_letter if recognized else None,
        "confidence": confidence,
        "recognized": recognized,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)