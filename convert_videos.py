"""Convert labeled J/Z video clips into motion landmark rows for train_model.py.

Expected input layout:

    video_folder/
      J/  one or more .avi (or other OpenCV-readable) clips
      Z/  one or more .avi (or other OpenCV-readable) clips

Each video becomes one CSV row containing 20 sampled frames, with 21 hand
landmarks (x, y, z) per frame, followed by the sign label.
"""

from __future__ import annotations
from video_sources import calculate_video_hash, make_source_record
from motion_dataset import MotionDataset

import math
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert J/Z videos with source tracking and deduplication"
    )
    parser.add_argument("video_folder", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/motion_landmarks.csv"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append only to a dataset with a matching source manifest",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check source hashes without writing files",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        help="Process at most this many input files per label",
    )
    args = parser.parse_args(argv)

    if args.sequence_length != SEQUENCE_LENGTH:
        parser.error(
            f"This project requires {SEQUENCE_LENGTH} frames per clip."
        )

    if args.limit_per_label is not None and args.limit_per_label < 1:
        parser.error("--limit-per-label must be at least 1.")

    video_folder = args.video_folder.expanduser().resolve()
    output_path = args.output.expanduser().absolute()

    if not video_folder.is_dir():
        parser.error(f"Video folder does not exist: {video_folder}")

    all_videos = find_videos(video_folder)
    if not all_videos:
        parser.error("No supported videos were found in J/ or Z/.")

    videos = []
    selected = {label: 0 for label in SIGNS}

    for path, label in all_videos:
        if (
            args.limit_per_label is None
            or selected[label] < args.limit_per_label
        ):
            videos.append((path, label))
            selected[label] += 1

    sources = {}
    unique_videos = []
    duplicate_inputs = 0
    conflicts = 0
    errors = 0

    for video_path, sign in videos:
        try:
            record = make_source_record(
                video_path, sign, video_folder
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"ERROR: {video_path.name}: {error}")
            errors += 1
            continue

        previous = sources.get(record["source_sha256"])

        if previous is None:
            sources[record["source_sha256"]] = record
            unique_videos.append((video_path, record))

        elif previous["label"] != sign:
            conflicts += 1
            print(
                "LABEL CONFLICT: "
                f"{record['source_path']} and "
                f"{previous['source_path']}"
            )

        else:
            duplicate_inputs += 1
            print(
                "DUPLICATE: "
                f"{record['source_path']} matches "
                f"{previous['source_path']}"
            )

    print("\n=== Video Source Check ===")
    print(f"Files found: {len(all_videos)}")
    print(f"Files selected: {len(videos)}")
    print(f"Unique file contents: {len(sources)}")
    print(f"Duplicate files: {duplicate_inputs}")
    print(f"Conflicting labels: {conflicts}")
    print(f"Unreadable or changed files: {errors}")

    if args.dry_run:
        print("Dry run complete. No files were written.")
        return 1 if conflicts or errors else 0

    if conflicts or errors:
        parser.error(
            "Source checks failed. No dataset changes were made."
        )

    if output_path.exists() and not args.append:
        parser.error(
            "Output already exists. Choose a NEW --output."
        )

    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "hand_landmarker.task"
    )
    normalizer_path = (
        Path(__file__).resolve().parent / "convert_images.py"
    )

    try:
        signature = {
            "sequence_length": SEQUENCE_LENGTH,
            "hand_model_sha256": calculate_video_hash(model_path),
            "normalizer_file_sha256": calculate_video_hash(
                normalizer_path
            ),
            "mediapipe_version": mp.__version__,
        }

        dataset = MotionDataset(
            output_path,
            CSV_HEADER,
            append=args.append,
            signature=signature,
        )

        pending = []
        existing_duplicates = 0

        for video_path, record in unique_videos:
            if dataset.contains(record):
                existing_duplicates += 1
                print(
                    f"ALREADY IN DATASET: {record['source_path']}"
                )
            else:
                pending.append((video_path, record))

        skipped = 0

        if pending:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            with mp.tasks.vision.HandLandmarker.create_from_options(
                options
            ) as hands:
                for index, (video_path, record) in enumerate(
                    pending, start=1
                ):
                    print(
                        f"[{index}/{len(pending)}] Processing "
                        f"{record['label']}: "
                        f"{record['source_path']}"
                    )

                    features = sample_video(
                        video_path,
                        hands,
                        SEQUENCE_LENGTH,
                    )

                    if features is None:
                        skipped += 1
                        continue

                    if (
                        len(features) != FEATURE_COUNT
                        or not all(
                            math.isfinite(float(value))
                            for value in features
                        )
                    ):
                        skipped += 1
                        print("SKIPPED: invalid feature values")
                        continue

                    if (
                        calculate_video_hash(video_path)
                        != record["source_sha256"]
                    ):
                        raise ValueError(
                            "A source video changed during conversion. "
                            "No new rows were saved."
                        )

                    record["sampled_frames"] = SEQUENCE_LENGTH
                    dataset.add(features, record)

        if (
            calculate_video_hash(model_path)
            != signature["hand_model_sha256"]
            or calculate_video_hash(normalizer_path)
            != signature["normalizer_file_sha256"]
        ):
            raise ValueError(
                "Conversion assets changed while processing. "
                "No new rows were saved."
            )

        added = dataset.save()

    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.error(str(error))

    print("\n=== Conversion Summary ===")
    print(f"Added rows: {added}")
    print(f"Duplicate input files: {duplicate_inputs}")
    print(f"Already in dataset: {existing_duplicates}")
    print(f"Skipped videos: {skipped}")
    print(f"Total dataset rows: {len(dataset.entries)}")

    if dataset.output.exists() and dataset.manifest.exists():
        print(f"CSV: {dataset.output}")
        print(f"Manifest: {dataset.manifest}")
    else:
        print("No dataset files were created.")

    if added == 0 and existing_duplicates == 0:
        print(
            "No usable new samples were converted.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())