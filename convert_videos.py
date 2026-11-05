"""Convert labeled J/Z video clips into motion landmark rows for train_model.py.

Expected input layout:

    video_folder/
      J/  one or more .avi (or other OpenCV-readable) clips
      Z/  one or more .avi (or other OpenCV-readable) clips

Each video becomes one CSV row containing 20 sampled frames, with 21 hand
landmarks (x, y, z) per frame, followed by the sign label.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from convert_images import normalize_landmarks


SEQUENCE_LENGTH = 20
SIGNS = ("J", "Z")
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
FEATURE_COUNT = SEQUENCE_LENGTH * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
CSV_HEADER = [f"f{i}" for i in range(FEATURE_COUNT)] + ["label"]


def find_videos(folder: Path) -> list[tuple[Path, str]]:
    """Find video files in J and Z subfolders (including nested folders)."""
    videos: list[tuple[Path, str]] = []
    for sign in SIGNS:
        sign_folder = folder / sign
        if not sign_folder.is_dir():
            print(f"Warning: no {sign}/ folder found at {sign_folder}")
            continue
        for path in sorted(sign_folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append((path, sign))
    return videos


def read_sampled_frames(video_path: Path, sequence_length: int) -> list:
    """Count decoded frames, then read evenly spaced samples sequentially.

    AVI frame-count metadata and random seeking can be unreliable. Two passes
    avoid relying on either and keep only the sampled frames in memory.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            print(f"Warning: could not open video: {video_path}")
            return []

        frame_count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frame_count += 1
    finally:
        cap.release()

    if frame_count <= 0:
        print(f"Warning: video has no readable frames: {video_path}")
        return []

    sample_indices = [round(i * (frame_count - 1) / max(sequence_length - 1, 1))
                      for i in range(sequence_length)]
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame_index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                print(f"Warning: failed reading a sampled frame: {video_path}")
                return []
            # Short clips can have repeated sample indices.
            while len(frames) < sequence_length and sample_indices[len(frames)] == frame_index:
                frames.append(frame)
            if len(frames) == sequence_length:
                break
    finally:
        cap.release()
    return frames


def sample_video(video_path: Path, hands, sequence_length: int) -> list[float] | None:
    """Sample evenly spaced frames and return flattened hand landmarks.

    If the hand is not detected in any sampled frame, the clip is skipped.
    The largest detected hand is selected when a frame contains multiple hands.
    """
    frames = read_sampled_frames(video_path, sequence_length)
    if len(frames) != sequence_length:
        return None
    flattened: list[float] = []

    for frame in frames:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            print(f"Warning: no hand detected in sampled frames; skipping: {video_path}")
            return None

        # Pick the detected hand with the largest image-space bounding box.
        hand_index = max(
            range(len(result.hand_landmarks)),
            key=lambda index: (
                max(point.x for point in result.hand_landmarks[index]) - min(point.x for point in result.hand_landmarks[index])
            ) * (
                max(point.y for point in result.hand_landmarks[index]) - min(point.y for point in result.hand_landmarks[index])
            ),
        )
        # Match the wrist-relative, handedness-aware features used at runtime.
        flattened.extend(normalize_landmarks(
            result.hand_landmarks[hand_index],
            result.handedness[hand_index][0].category_name,
        ))

    return flattened


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert J/ and Z/ video folders into data/motion_landmarks.csv"
    )
    parser.add_argument(
        "video_folder",
        type=Path,
        help="Folder containing J and Z subfolders of video clips",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/motion_landmarks.csv"),
        help="Output CSV path (default: data/motion_landmarks.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append clips to an existing CSV instead of refusing to overwrite it",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Frames sampled per clip (must match the model training sequence length)",
    )
    args = parser.parse_args()

    if args.sequence_length != SEQUENCE_LENGTH:
        parser.error(
            f"this project currently expects {SEQUENCE_LENGTH} frames per clip; "
            "change train_model.py and the runtime model input together before using another value"
        )

    video_folder = args.video_folder.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not video_folder.is_dir():
        parser.error(f"video folder does not exist: {video_folder}")

    videos = find_videos(video_folder)
    if not videos:
        parser.error(f"no supported video files found under {video_folder}/J or {video_folder}/Z")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.append:
        parser.error(
            f"{output_path} already exists. Use --append to add to it, "
            "or move/delete it first if you intend to regenerate the dataset."
        )

    mode = "a" if args.append and output_path.exists() else "w"
    needs_header = mode == "w" or output_path.stat().st_size == 0
    written = 0
    skipped = 0

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(
            Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        )),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as hands, output_path.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if needs_header:
            writer.writerow(CSV_HEADER)

        for index, (video_path, sign) in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] Processing {sign}: {video_path.name}")
            features = sample_video(video_path, hands, args.sequence_length)
            if features is None:
                skipped += 1
                continue
            if len(features) != FEATURE_COUNT:
                print(f"Warning: unexpected feature count for {video_path}; skipping")
                skipped += 1
                continue
            writer.writerow([*features, sign])
            written += 1

    print(f"Done. Added {written} clips; skipped {skipped}. CSV: {output_path}")
    if written == 0:
        print("No usable clips were added. Check that each video shows a visible hand.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
