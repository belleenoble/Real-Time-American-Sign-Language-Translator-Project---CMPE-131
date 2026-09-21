from collections import deque

import time

import cv2
import joblib
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/hand_landmarker.task"
STATIC_MODEL_PATH = "models/static_sign_classifier.joblib"
MOTION_MODEL_PATH = "models/motion_sign_classifier.joblib"

# Minimum prediction confidence required before a sign is trusted.
# Anything below this is treated as "unrecognized" (User Story 2).
CONFIDENCE_THRESHOLD = 0.6

<<<<<<< HEAD
=======
MOTION_CONFIDENCE_THRESHOLD = 0.9
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5
#for static signs
# how many consecutive frames must agree before a letter is committed
# to the word. Higher = more stable but slower to react.
STABILITY_FRAMES = 25

#for motion signs
#needs to match SEQUENCE_LENGTH in collect_data.py
MOTION_SEQUENCE_LENGTH = 20 

#motion signs get fewer, noiser windows than static so it needs to be more stable
<<<<<<< HEAD
MOTION_STABILITY_FRAMES = 5
=======
MOTION_STABILITY_FRAMES = 8

MOTION_CHECK_INTERVAL = 4
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5

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
        f"No motion model found at {'MOTION_MODEL_PATH'}. Motion signs will not be recognized."
    )

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
motion_frame_buffer = deque(maxlen=MOTION_SEQUENCE_LENGTH)
recent_motion_predictions =[]

current_word = ""
sentence =""
<<<<<<< HEAD
=======
frame_count = 0
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5
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

<<<<<<< HEAD
=======
            frame_count += 1
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5
            static_sign = None
            confidence = 0.0
            motion_sign = None
            motion_confidence =0.0

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

                if static_confidence >= CONFIDENCE_THRESHOLD:
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
                        motion_frame_buffer.clear() #clear motion buffer when a static sign is committed
                        recent_motion_predictions = []
                else:
                    recent_static_predictions = []

                #motion sign preritctions
                if motion_model is not None:
                    motion_frame_buffer.append(features)

<<<<<<< HEAD
                    if len(motion_frame_buffer) >= MOTION_SEQUENCE_LENGTH:
=======
                    if (len(motion_frame_buffer) >= MOTION_SEQUENCE_LENGTH 
                    and frame_count % MOTION_CHECK_INTERVAL == 0 
                    ):
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5
                        flat_seqeunce = np.concatenate(motion_frame_buffer)
                        motion_probabilities = motion_model.predict_proba([flat_seqeunce])[0]
                        best_motion_index = np.argmax(motion_probabilities)
                        motion_sign = motion_model.classes_[best_motion_index]
                        motion_confidence = motion_probabilities[best_motion_index]

<<<<<<< HEAD
                        if motion_confidence >= CONFIDENCE_THRESHOLD:
=======
                        if motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5
                            recent_motion_predictions.append(motion_sign)
                            recent_motion_predictions = (recent_motion_predictions[-MOTION_STABILITY_FRAMES:])

                            is_motion_stable = (
                                len(recent_motion_predictions) == MOTION_STABILITY_FRAMES
                                and len(set(recent_motion_predictions)) == 1
                            )

                            if (is_motion_stable and motion_sign != last_committed_sign):
                                current_word += motion_sign
                                last_committed_sign = motion_sign
                                recent_motion_predictions = []
                                motion_frame_buffer.clear() #clear motion buffer when a motion sign is committed
                                recent_static_predictions = []

                        else:
                            recent_motion_predictions = []

            else:
                frames_since_hand_seen += 1
                recent_static_predictions = []
                motion_frame_buffer.clear()
                recent_motion_predictions = []

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
<<<<<<< HEAD
                motion_color = (128, 128, 128)
=======
                motion_color = (0, 255, 0) if motion_confidence >= CONFIDENCE_THRESHOLD else (0,0,255)
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5

            elif motion_sign is not None:
                motion_status = (
                    f"Motion: {motion_sign} "
                    f"({motion_confidence * 100:.0f}% confidence)"
                )
<<<<<<< HEAD
                motion_color = (0, 255, 0) if motion_confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
=======
                motion_color = (128, 128, 128) if motion_confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
>>>>>>> e72eee194c568a6fae26e066d5602c92a33e4cc5
            else:
                buffered = len(motion_frame_buffer)
                motion_status = (
                    f"Motion: {buffered}/{MOTION_SEQUENCE_LENGTH} frames buffered"
                )
                motion_color = (255, 255, 0)  # Yellow for buffering

            cv2.putText(
                frame, motion_status, (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2
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
                current_word = ""
                last_committed_sign = None
                recent_static_predictions = []
                motion_frame_buffer.clear()
                recent_motion_predictions = []

            #option to let words break manually (for sentences)

            if key == ord (" ") and current_word:
                sentence += current_word + " "
                current_word = ""
                last_committed_sign = None

finally:
    camera.release()
    cv2.destroyAllWindows()