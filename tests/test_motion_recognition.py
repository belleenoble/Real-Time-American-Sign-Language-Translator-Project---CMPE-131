import unittest
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import mediapipe as mp

from motion_recognition import MotionRecognizer, resample_motion


class FakeModel:
    classes_ = np.array(["J", "Z"])

    def __init__(self, confidence=0.95):
        self.confidence = confidence
        self.inputs = []

    def predict_proba(self, rows):
        self.inputs.append(np.array(rows))
        return np.array([[self.confidence, 1 - self.confidence]])


class MotionTests(unittest.TestCase):
    def test_time_sampling_covers_complete_gesture_at_different_frame_rates(self):
        for fps in (20, 30, 60):
            times = np.linspace(0, 2, 2 * fps + 1)
            samples = [(t, np.full(63, t)) for t in times]
            result = resample_motion(samples).reshape(20, 63)
            np.testing.assert_allclose(result[:, 0], np.linspace(0, 2, 20), atol=1 / fps)
            self.assertEqual(result[0, 0], 0)
            self.assertEqual(result[-1, 0], 2)

    def test_manual_capture_accepts_one_complete_gesture(self):
        model = FakeModel()
        motion = MotionRecognizer(model)
        motion.toggle_recording(0)
        for i in range(40):
            self.assertIsNone(motion.observe(np.full(63, i), i * 0.05))
        self.assertEqual(model.inputs, [])
        self.assertEqual(motion.toggle_recording(2), "J")
        self.assertEqual(model.inputs[0].shape, (1, 1260))
        self.assertEqual(model.inputs[0][0, -1], 39)
        status = motion.status
        motion.observe(np.zeros(63), 2.05)
        self.assertEqual(motion.status, status)

    def test_low_confidence_is_visible_but_not_accepted(self):
        motion = MotionRecognizer(FakeModel(0.65))
        motion.toggle_recording(0)
        for i in range(25):
            motion.observe(np.zeros(63), i * 0.05)
        self.assertIsNone(motion.toggle_recording(1.25))
        self.assertIn("65%", motion.status)
        self.assertIn("below threshold", motion.status)
        self.assertFalse(motion.accepted)

    def test_short_capture_does_not_call_model(self):
        model = FakeModel()
        motion = MotionRecognizer(model)
        motion.toggle_recording(0)
        motion.observe(np.zeros(63), 0.05)
        self.assertIsNone(motion.toggle_recording(0.1))
        self.assertIn("too short", motion.status)
        self.assertEqual(model.inputs, [])

    def test_brief_tracking_loss_keeps_history_but_long_loss_cancels(self):
        motion = MotionRecognizer(FakeModel())
        motion.toggle_recording(0)
        motion.observe(np.zeros(63), 0.05)
        motion.observe(None, 0.1)
        self.assertEqual(len(motion.samples), 1)
        self.assertTrue(motion.recording)
        motion.observe(None, 0.5)
        self.assertEqual(len(motion.samples), 0)
        self.assertFalse(motion.recording)
        self.assertIn("cancelled", motion.status)

    def test_automatic_prediction_waits_for_time_window_and_stability(self):
        model = FakeModel()
        motion = MotionRecognizer(model)
        for i in range(30):
            self.assertIsNone(motion.observe(np.zeros(63), i * 0.05))
        self.assertEqual(model.inputs, [])
        results = [motion.observe(np.zeros(63), i * 0.05) for i in range(30, 60)]
        self.assertEqual([r for r in results if r], ["J"])
        self.assertGreaterEqual(len(model.inputs), 5)

    def test_reset_cancels_capture_and_discards_history(self):
        motion = MotionRecognizer(FakeModel())
        motion.toggle_recording(0)
        motion.observe(np.zeros(63), 0.05)
        motion.reset()
        self.assertFalse(motion.recording)
        self.assertEqual(len(motion.samples), 0)


class WebcamLoopTests(unittest.TestCase):
    def run_camera_loop(self, keys):
        """Exercise the real script without opening a camera or GUI."""
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = lambda: (True, np.zeros((480, 640, 3), dtype=np.uint8))
        detector = MagicMock()
        detector.__enter__.return_value = detector
        detector.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=i / 25, y=i / 30, z=0) for i in range(21)]],
            handedness=[[SimpleNamespace(category_name="Right")]],
        )
        static = FakeModel()
        static.classes_ = np.array(["A", "B"])
        with (
            patch("cv2.VideoCapture", return_value=camera),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch("cv2.waitKey", side_effect=keys),
            patch("time.perf_counter", side_effect=[i * 0.05 for i in range(200)]),
            patch("joblib.load", side_effect=[static, FakeModel()]),
            patch.object(mp.tasks.vision.HandLandmarker, "create_from_options", return_value=detector),
        ):
            state = runpy.run_path(str(Path(__file__).resolve().parents[1] / "recognize.py"))
        camera.release.assert_called_once()
        return state

    def test_static_commit_does_not_prevent_automatic_motion(self):
        state = self.run_camera_loop([-1] * 79 + [27])
        self.assertTrue(state["current_word"].startswith("AJ"))

    def test_manual_capture_suppresses_static_and_accepts_motion(self):
        state = self.run_camera_loop([ord("m")] + [-1] * 43 + [ord("m"), 27])
        self.assertEqual(state["current_word"], "J")


if __name__ == "__main__":
    unittest.main()
