"""Hash/metadata tests with synthetic files, not real ASL videos."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from video_sources import calculate_video_hash, make_source_record


class VideoSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

        self.root = Path(self.temporary.name)
        self.video_root = self.root / "videos"

        (self.video_root / "J").mkdir(parents=True)
        (self.video_root / "Z").mkdir()

    def create_file(self, relative_path, content):
        path = self.video_root / relative_path
        path.write_bytes(content)
        return path

    def test_hash_matches_sha256(self):
        path = self.create_file("J/one.avi", b"sample video")
        expected = hashlib.sha256(b"sample video").hexdigest()

        self.assertEqual(calculate_video_hash(path), expected)

    def test_identical_bytes_have_same_hash_despite_different_names(self):
        first = self.create_file("J/one.avi", b"same content")
        second = self.create_file("J/renamed.avi", b"same content")

        self.assertEqual(
            calculate_video_hash(first),
            calculate_video_hash(second),
        )

    def test_different_bytes_have_different_hashes(self):
        first = self.create_file("J/one.avi", b"first content")
        second = self.create_file("Z/two.avi", b"second content")

        self.assertNotEqual(
            calculate_video_hash(first),
            calculate_video_hash(second),
        )

    def test_source_record_contains_relative_path_and_label(self):
        path = self.create_file("J/one.avi", b"sample video")
        record = make_source_record(path, "J", self.video_root)

        self.assertEqual(record["source_path"], "J/one.avi")
        self.assertEqual(record["label"], "J")
        self.assertEqual(record["source_bytes"], len(b"sample video"))
        self.assertEqual(
            record["source_sha256"],
            calculate_video_hash(path),
        )

    def test_invalid_label_is_rejected(self):
        path = self.create_file("J/one.avi", b"sample video")

        with self.assertRaises(ValueError):
            make_source_record(path, "A", self.video_root)

    def test_missing_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            make_source_record(
                self.video_root / "J/missing.avi",
                "J",
                self.video_root,
            )

    def test_file_outside_input_folder_is_rejected(self):
        outside = self.root / "outside.avi"
        outside.write_bytes(b"outside content")

        with self.assertRaises(ValueError):
            make_source_record(outside, "J", self.video_root)

    def test_source_changed_during_hashing_is_rejected(self):
        path = self.create_file("J/one.avi", b"original")

        def change_during_hash(video_path):
            Path(video_path).write_bytes(b"changed and longer content")
            return "0" * 64

        with patch(
            "video_sources.calculate_video_hash",
            side_effect=change_during_hash,
        ):
            with self.assertRaises(RuntimeError):
                make_source_record(path, "J", self.video_root)


if __name__ == "__main__":
    unittest.main()
    