

import time

import cv2
import joblib
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/hand_landmarker.task"
STATIC_MODEL_PATH = "models/static_sign_classifier.joblib"

# Minimum prediction confidence required before a sign is trusted.
# Anything below this is treated as "unrecognized" (User Story 2).
CONFIDENCE_THRESHOLD = 0.6

# How many consecutive frames must agree before a letter is committed
# to the word. Higher = more stable but slower to react.
STABILITY_FRAMES = 15

# How many consecutive no-hand frames before we allow the same letter
# to be signed again (e.g. the double L in "HELLO").
HAND_LOST_RESET_FRAMES = 10


def normalize_landmarks(landmarks, hand_name):
    coordinates = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float32
    )

    # Move wrist landmark to the origin
    coordinates -= coordinates[0]

    # Mirror left hands so both hands have a similar orientation
    if hand_name == "Left":
        coordinates[:, 0] *= -1

    # Make the data independent of hand size and camera distance
    scale = np.max(np.linalg.norm(coordinates, axis=1))

    if scale > 0:
        coordinates /= scale

    return coordinates.flatten()


static_model = joblib.load(STATIC_MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

# Recognition / word-building state
recent_predictions = []
committed_word = ""
last_committed_sign = None
frames_since_hand_seen = 0

start_time = time.perf_counter()
last_timestamp = -1

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = int(
                (time.perf_counter() - start_time) * 1000
            )
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = landmarker.detect_for_video(mp_image, timestamp)

            predicted_sign = None
            confidence = 0.0

            if result.hand_landmarks:
                frames_since_hand_seen = 0

                landmarks = result.hand_landmarks[0]
                hand_name = result.handedness[0][0].category_name

                for point in landmarks:
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                features = normalize_landmarks(landmarks, hand_name)

                probabilities = static_model.predict_proba([features])[0]
                best_index = np.argmax(probabilities)
                predicted_sign = static_model.classes_[best_index]
                confidence = probabilities[best_index]

                if confidence >= CONFIDENCE_THRESHOLD:
                    recent_predictions.append(predicted_sign)
                    recent_predictions = recent_predictions[-STABILITY_FRAMES:]

                    is_stable = (
                        len(recent_predictions) == STABILITY_FRAMES
                        and len(set(recent_predictions)) == 1
                    )

                    if is_stable and predicted_sign != last_committed_sign:
                        committed_word += predicted_sign
                        last_committed_sign = predicted_sign
                else:
                    recent_predictions = []

            else:
                frames_since_hand_seen += 1
                recent_predictions = []

                if frames_since_hand_seen >= HAND_LOST_RESET_FRAMES:
                    last_committed_sign = None

            # --- on-screen display ---
            cv2.putText(
                frame,
                f"Word: {committed_word}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

            if predicted_sign is not None:
                status = (
                    f"Seeing: {predicted_sign} "
                    f"({confidence * 100:.0f}% confidence)"
                )
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            else:
                status = "No hand detected"
                color = (0, 0, 255)

            cv2.putText(
                frame, status, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            cv2.putText(
                frame,
                "Press C to clear word, ESC to quit",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            cv2.imshow("ASL Live Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key exits data collection; Q remains available as a sign:
                break

            if key == ord("c"):
                committed_word = ""
                last_committed_sign = None
                recent_predictions = []

finally:
    camera.release()
    cv2.destroyAllWindows()