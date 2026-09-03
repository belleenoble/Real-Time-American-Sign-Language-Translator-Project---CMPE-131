import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/hand_landmarker.task"
DATA_PATH = "data/landmarks.csv"

LETTERS = ["A", "B", "C", "D", "E"]
SAMPLES_PER_LETTER = 200

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


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

file_exists = os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

active_letter = None
sample_count = 0
frame_count = 0
start_time = time.perf_counter()
last_timestamp = -1

try:
    with open(DATA_PATH, "a", newline="") as data_file:
        writer = csv.writer(data_file)

        if not file_exists:
            header = ["label"]

            for landmark_number in range(21):
                header.extend([
                    f"x{landmark_number}",
                    f"y{landmark_number}",
                    f"z{landmark_number}"
                ])

            writer.writerow(header)

        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
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

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

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

                    # Save one sample every two frames
                    if active_letter and frame_count % 2 == 0:
                        features = normalize_landmarks(
                            landmarks,
                            hand_name
                        )

                        writer.writerow([active_letter, *features])
                        data_file.flush()
                        sample_count += 1

                        if sample_count >= SAMPLES_PER_LETTER:
                            print(
                                f"Finished collecting {active_letter}"
                            )
                            active_letter = None
                            sample_count = 0

                cv2.putText(
                    frame,
                    "Press A, B, C, D, or E to collect",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                if active_letter:
                    message = (
                        f"Collecting {active_letter}: "
                        f"{sample_count}/{SAMPLES_PER_LETTER}"
                    )
                    color = (0, 255, 0)
                else:
                    message = "Waiting for a letter selection"
                    color = (0, 255, 255)

                cv2.putText(
                    frame,
                    message,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2
                )

                cv2.imshow("ASL Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key != 255:
                    selected = chr(key).upper()

                    if selected in LETTERS and active_letter is None:
                        active_letter = selected
                        sample_count = 0
                        print(f"Collecting letter {selected}...")

finally:
    camera.release()
    cv2.destroyAllWindows()