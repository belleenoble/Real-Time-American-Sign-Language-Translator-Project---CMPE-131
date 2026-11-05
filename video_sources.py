"""Source identifiers for reproducible video conversion."""

import hashlib
from pathlib import Path


def calculate_video_hash(video_path):
    """Identify byte-identical files, even after renaming."""
    video_path = Path(video_path)
    checksum = hashlib.sha256()

    with video_path.open("rb") as video:
        for block in iter(lambda: video.read(1024 * 1024), b""):
            checksum.update(block)

    return checksum.hexdigest()


def make_source_record(video_path, label, input_root):
    """Collect source metadata without changing the video."""
    video_path = Path(video_path).expanduser().absolute()
    input_root = Path(input_root).expanduser().absolute()

    if label not in {"J", "Z"}:
        raise ValueError("The video label must be J or Z.")

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        relative_path = video_path.relative_to(input_root)
    except ValueError as error:
        raise ValueError(
            "The video must be inside the input folder."
        ) from error

    before = video_path.stat()
    checksum = calculate_video_hash(video_path)
    after = video_path.stat()

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(
            "The video changed while being read. Retry after saving it."
        )

    return {
        "source_path": relative_path.as_posix(),
        "label": label,
        "source_bytes": after.st_size,
        "source_sha256": checksum,
    }