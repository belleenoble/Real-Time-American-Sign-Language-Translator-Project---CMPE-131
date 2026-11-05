import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from motion_recognition import MotionRecognizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models/hand_landmarker.task")
STATIC_MODEL_PATH = BASE_DIR / "models/static_sign_classifier.joblib"
MOTION_MODEL_PATH = BASE_DIR / "models/motion_sign_classifier.joblib"

# Minimum prediction confidence required before a sign is trusted.
# Anything below this is treated as "unrecognized" (User Story 2).
CONFIDENCE_THRESHOLD = 0.6

MOTION_CONFIDENCE_THRESHOLD = 0.9
#for static signs
# how many consecutive frames must agree before a letter is committed
# to the word. Higher = more stable but slower to react.
STABILITY_FRAMES = 25

WORD_BREAK_FRAMES = 30  # how many frames of no hand before the current word is reset

#for double letters 
LETTER_RESET_FRAMES = 10


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

#9/18/26 testing a motion model for J and Z 

try: 
    motion_model =joblib.load(MOTION_MODEL_PATH)
    print(f"Loaded motion model: {list(motion_model.classes_)}")
except FileNotFoundError:
    motion_model = None
    print(
        f"No motion model found at {MOTION_MODEL_PATH}. Motion signs will not be recognized."
    )

motion = MotionRecognizer(motion_model, MOTION_CONFIDENCE_THRESHOLD) if motion_model is not None else None
print("Press M to start a J/Z gesture, then M again to classify it. C clears; ESC quits.")

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
recent_static_predictions = []

current_word = ""
sentence =""
last_committed_sign = None
frames_since_hand_seen = 0
word_break_triggered =False

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

            static_sign = None
            static_confidence = 0.0
            now = timestamp / 1000.0

            if result.hand_landmarks:
                frames_since_hand_seen = 0
                word_break_triggered = False

                landmarks = result.hand_landmarks[0]
                hand_name = result.handedness[0][0].category_name

                for point in landmarks:
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                features = normalize_landmarks(landmarks, hand_name)

                #static sign predictions

                static_probabilities = static_model.predict_proba([features])[0]
                best_static_index = np.argmax(static_probabilities)
                static_sign = static_model.classes_[best_static_index]
                static_confidence = static_probabilities[best_static_index]

                motion_sign = motion.observe(features, now) if motion else None
                if motion_sign is not None:
                    if motion_sign != last_committed_sign:
                        current_word += motion_sign
                        last_committed_sign = motion_sign
                    recent_static_predictions = []

                # Manual gesture capture suppresses static-letter commits.
                # Static predictions must never erase the motion history.
                elif not (motion and motion.recording) and static_confidence >= CONFIDENCE_THRESHOLD:
                    recent_static_predictions.append(static_sign)
                    recent_static_predictions = recent_static_predictions[-STABILITY_FRAMES:]

                    is_static_stable = (
                        len(recent_static_predictions) == STABILITY_FRAMES
                        and len(set(recent_static_predictions)) == 1
                    )

                    if is_static_stable and static_sign != last_committed_sign:
                        current_word += static_sign
                        last_committed_sign = static_sign
                        recent_static_predictions = []
                else:
                    recent_static_predictions = []

            else:
                frames_since_hand_seen += 1
                recent_static_predictions = []
                if motion:
                    motion.observe(None, now)

                if frames_since_hand_seen >= LETTER_RESET_FRAMES:
                    last_committed_sign = None

            #word breaks (spaces) are triggered by a period of no hand detection this addes the space between words/phrases

                if (
                    frames_since_hand_seen >= WORD_BREAK_FRAMES 
                    and current_word 
                    and not word_break_triggered
                ):

                    sentence += current_word + " "
                    current_word = ""
                    word_break_triggered = True

            # --- on-screen display ---
            cv2.putText(
                frame,
                f"Sentence: {sentence}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Current word: {current_word}",
                (10,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

            if static_sign is not None:
                static_status = (
                    f"Seeing: {static_sign} "
                    f"({static_confidence * 100:.0f}% confidence)"
                )
                static_color = (0, 255, 0) if static_confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            else:
                static_status = "No hand detected"
                static_color = (0, 0, 255)

            cv2.putText(
                frame, static_status, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, static_color, 2
            )

            if motion_model is None:
                motion_status = "No motion model loaded"
                motion_color = (128, 128, 128)

            else:
                motion_status = motion.status
                motion_color = (0, 255, 0) if motion.accepted else (0, 200, 255)

            cv2.putText(
                frame, motion_status, (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, min(0.6, (width - 20) / max(1, cv2.getTextSize(motion_status, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0])), motion_color, 2
            )
            
            cv2.putText(
                frame,
                "M: start/finish J/Z gesture | C: clear | ESC: quit",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            cv2.imshow("ASL Live Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key exits data collection; Q remains available as a sign:
                break

            if key in (ord("c"), ord("C")):
                current_word = ""
                last_committed_sign = None
                recent_static_predictions = []
                if motion:
                    motion.reset()

            if key in (ord("m"), ord("M")) and motion:
                motion_sign = motion.toggle_recording(now)
                recent_static_predictions = []
                if motion_sign is not None:
                    # Each explicit recording is a separate intentional gesture.
                    current_word += motion_sign
                    last_committed_sign = motion_sign

            #option to let words break manually (for sentences)

            if key == ord (" ") and current_word:
                sentence += current_word + " "
                current_word = ""
                last_committed_sign = None

finally:
    camera.release()
    cv2.destroyAllWindows()
