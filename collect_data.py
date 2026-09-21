import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/hand_landmarker.task"
STATIC_DATA_PATH = "data/landmarks.csv"
MOTION_DATA_PATH = "data/motion_landmarks.csv" #added it for J and Z (only motion signs) 

#Static signs are everything)
STATIC_LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  
DIGITS = [str(d) for d in range(10)]
STATIC_SIGNS = STATIC_LETTERS + DIGITS
SAMPLES_PER_SIGN = 200

#motion signs are J and Z (only motion signs) for the alphabet
MOTION_SIGNS = ["J", "Z"]
SEQUENCE_LENGTH = 20
SEQUENCES_PER_SIGN = 100

os.makedirs("data", exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


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

def static_header():
    header = ["label"]

    for landmark_number in range(21):
        header.extend([
            f"x{landmark_number}",
            f"y{landmark_number}",
            f"z{landmark_number}"
        ])

    return header

def motion_header():
    header =["label"]

    for frame_number in range(SEQUENCE_LENGTH):
        for landmark_number in range(21):
            header.extend([
                f"frame{frame_number}_x{landmark_number}",
                f"frame{frame_number}_y{landmark_number}",
                f"frame{frame_number}_z{landmark_number}",
            ])

    return header

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

#needed to add motion file for other letters (j and z)
static_file_exists = os.path.exists(STATIC_DATA_PATH) and os.path.getsize(STATIC_DATA_PATH) > 0
motion_file_exists = os.path.exists(MOTION_DATA_PATH) and os.path.getsize(MOTION_DATA_PATH) > 0

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

#static collection state
active_static_sign = None
static_sample_count = 0
frame_count = 0

#motion collection state 

active_motion_sign = None
motion_sequence_count = 0
current_sequence = []
recording_sequence = False

start_time = time.perf_counter()
last_timestamp = -1

#once again needed to add both motion and static
static_file = open(STATIC_DATA_PATH, "a", newline="")
motion_file = open(MOTION_DATA_PATH, "a", newline="")

try:
    static_writer = csv.writer(static_file)
    motion_writer = csv.writer(motion_file)

    if not static_file_exists:
        static_writer.writerow(static_header())

    if not motion_file_exists:
        motion_writer.writerow(motion_header())

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame!")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp = int((time.perf_counter() - start_time) * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = landmarker.detect_for_video(mp_image, timestamp)

            frame_count += 1

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                hand_name = result.handedness[0][0].category_name

                height, width, _ = frame.shape

                # Draw the landmark dots
                for point in landmarks:
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                features = normalize_landmarks(landmarks, hand_name)

                # static sign collection: making it one sample for every two frames to avoid duplicates and to give the user time to change the sign

                if active_static_sign and frame_count % 2 == 0:
                    static_writer.writerow([active_static_sign, *features])
                    static_file.flush()
                    static_sample_count += 1

                    if static_sample_count >= SAMPLES_PER_SIGN:
                        print(f"Finished collecting {active_static_sign}")
                        active_static_sign = None
                        static_sample_count = 0

                #motion sign collection: making it one sample for every two frame to avoid duplicates and to give the user time to change the sign
                if recording_sequence:
                    current_sequence.append(features)

                    if len(current_sequence) >= SEQUENCE_LENGTH:
                        motion_writer.writerow([active_motion_sign, *np.array(current_sequence).flatten()])
                        motion_file.flush()
                        motion_sequence_count += 1
                        current_sequence = []

                        if motion_sequence_count >= SEQUENCES_PER_SIGN:
                            print(f"Finished collection '{active_motion_sign}'")
                            active_motion_sign = None
                            motion_sequence_count = 0
                            recording_sequence = False

            #this sections should be for the instructions on the user interface
            cv2.putText(
                frame,
                "Static signs: press A-Y or 0-9 (except J and Z) to collect samples",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            #adding motion signs instructions here
            cv2.putText(
                frame, "Motion signs: press J or Z, the SPACE per repetition to collect samples", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
            cv2.putText(
                frame,
                "Press ESC to quit",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if active_static_sign:
                message = (
                    f"Collecting {active_static_sign}: "
                    f"{static_sample_count}/{SAMPLES_PER_SIGN}"
                )
                color = (0, 255, 0)
            elif active_motion_sign:
                status = "Recording" if recording_sequence else "Waiting for SPACE"
                message = (
                    f"Collecting '{active_motion_sign}': "
                    f"{motion_sequence_count}/{SEQUENCES_PER_SIGN} "
                    f"sequences ({status})"
            )
                color = (0, 255, 255)

            else:
                message = "Waiting for a sign selection..."
                color = (255, 255, 0)

            cv2.putText(
                frame,
                message,
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key exits data collection; Q remains available as a sign
                break

            if key != 255:
                selected = chr(key).upper() if key < 128 else ""

                no_active_sign = (active_static_sign is None and active_motion_sign is None)

                if selected in STATIC_LETTERS and no_active_sign:
                    active_static_sign = selected
                    static_sample_count = 0
                    print(f"Collecting static sign {selected}...")

                elif selected in MOTION_SIGNS and no_active_sign:
                    active_motion_sign = selected
                    motion_sequence_count = 0
                    current_sequence = []
                    recording_sequence = False
                    print(f"Collecting motion sign '{selected}'..."
                            f"Press SPACE to record each repetition."
                    )

                elif (
                    key == ord(" ") and active_motion_sign and not recording_sequence
                ):
                    current_sequence = []
                    recording_sequence = True
                    print(f"Recording sequence for " f" '{active_motion_sign}'...")
finally:
    camera.release()
    cv2.destroyAllWindows()
    static_file.close()
    motion_file.close()
