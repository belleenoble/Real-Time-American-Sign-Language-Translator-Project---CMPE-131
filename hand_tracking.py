import time
import cv2
import mediapipe as mp


# Connections between MediaPipe's 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)
]

MODEL_PATH = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

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

            # Convert OpenCV BGR image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = int(
                (time.perf_counter() - start_time) * 1000
            )

            # MediaPipe requires increasing timestamps
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            height, width, _ = frame.shape

            for hand_number, landmarks in enumerate(
                result.hand_landmarks
            ):
                points = []

                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append((x, y))

                # Draw connections first
                for start_index, end_index in HAND_CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start_index],
                        points[end_index],
                        (0, 255, 0),
                        2
                    )

                # Draw all 21 landmarks
                for point in points:
                    cv2.circle(
                        frame,
                        point,
                        5,
                        (0, 0, 255),
                        -1
                    )

                # Display Left or Right
                hand_name = result.handedness[
                    hand_number
                ][0].category_name

                text_x = min(point[0] for point in points)
                text_y = max(
                    30,
                    min(point[1] for point in points) - 15
                )

                cv2.putText(
                    frame,
                    hand_name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                frame,
                "Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("MediaPipe Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()