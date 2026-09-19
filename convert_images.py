"""Convert labeled ASL images to MediaPipe landmarks using multiple CPU workers."""

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


_worker_landmarker = None


def normalize_landmarks(landmarks, hand_name):
    coordinates = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float32,
    )
    coordinates -= coordinates[0]

    if hand_name == "Left":
        coordinates[:, 0] *= -1

    scale = np.max(np.linalg.norm(coordinates, axis=1))
    if scale > 0:
        coordinates /= scale

    return coordinates.flatten()


def make_header():
    columns = ["label"]
    for landmark_number in range(21):
        columns.extend([
            f"x{landmark_number}",
            f"y{landmark_number}",
            f"z{landmark_number}",
        ])
    return columns


def initialize_worker(model_path):
    """Create one MediaPipe model in each worker process."""
    global _worker_landmarker

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _worker_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)


def process_batch(image_paths):
    rows = []
    skipped = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )
        result = _worker_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            skipped += 1
            continue

        landmarks = result.hand_landmarks[0]
        hand_name = result.handedness[0][0].category_name
        features = normalize_landmarks(landmarks, hand_name)
        label = image_path.parent.name.upper()
        rows.append([label, *features])

    return rows, skipped

def show_progress(done, total):
    width = 40
    ratio = done / total if total else 1
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)

    print(
        f"\r[{bar}] {ratio * 100:6.2f}% ({done}/{total})",
        end="",
        flush=True,
    )

def main():
    parser = argparse.ArgumentParser(
        description="Convert ASL images into the project's landmark CSV format."
    )
    parser.add_argument("input_dir", help="Folder containing one subfolder per label")
    parser.add_argument(
        "--output",
        default="data/landmarks.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU processes; use 6 or 8 on an 8-core computer",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a previous completed conversion",
    )
    parser.add_argument(
        "--include-motion-letters",
        action="store_true",
        help="Include J and Z as static classes",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    model_path = "models/hand_landmarker.task"
    if not os.path.exists(model_path):
        raise SystemExit(f"Missing {model_path}. Run this from the project folder.")

    completion_marker = args.output + ".done"
    if os.path.exists(completion_marker) and not args.force:
        print(f"Conversion already completed: {args.output}")
        print("Use --force only if you intentionally want to overwrite it.")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        path
        for path in Path(args.input_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        raise SystemExit("No images found. Check the input folder path.")

    if not args.include_motion_letters:
        image_paths = [
            path for path in image_paths
            if path.parent.name.upper() not in {"J", "Z"}
        ]

    if not image_paths:
        raise SystemExit("No usable images found after filtering labels.")

    output_directory = os.path.dirname(args.output)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    converted = 0
    skipped = 0

    with open(args.output, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(make_header())

        if args.workers == 1:
            initialize_worker(model_path)
            rows, batch_skipped = process_batch(image_paths)
            writer.writerows(rows)
            converted = len(rows)
            skipped = batch_skipped
            show_progress(converted + skipped, len(image_paths))
        else:
            worker_count = min(args.workers, len(image_paths))
            batch_size = 500
            batches = [
                image_paths[start:start + batch_size]
                for start in range(0, len(image_paths), batch_size)
            ]

            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=initialize_worker,
                initargs=(model_path,),
            ) as executor:
                futures = [
                    executor.submit(process_batch, batch)
                    for batch in batches
                ]

                for future in as_completed(futures):
                    rows, batch_skipped = future.result()
                    writer.writerows(rows)
                    converted += len(rows)
                    skipped += batch_skipped
                    show_progress(converted + skipped, len(image_paths))
    print()
    with open(completion_marker, "w") as marker_file:
        marker_file.write(
            f"Converted {converted} samples; skipped {skipped} images.\n"
        )

    print(f"Saved {converted} samples to {args.output}; skipped {skipped} images.")
    print("Conversion finished. The program will now exit.")


if __name__ == "__main__":
    main()
