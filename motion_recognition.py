"""Timed motion sampling and explicit whole-gesture capture for live recognition."""

from collections import deque

import numpy as np


SEQUENCE_LENGTH = 20


def resample_motion(samples, start=None):
    """Select 20 observations evenly in time, matching video conversion."""
    times = np.array([sample[0] for sample in samples])
    targets = np.linspace(times[0] if start is None else start, times[-1], SEQUENCE_LENGTH)
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    indices = np.where(targets - times[left] <= times[right] - targets, left, right)
    return np.concatenate([samples[index][1] for index in indices])


class MotionRecognizer:
    def __init__(self, model, confidence_threshold=0.9, window_seconds=1.5):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.window_seconds = window_seconds
        self.reset()

    def reset(self):
        self.samples = deque()
        self.recording = False
        self.last_seen = None
        self.last_check = float("-inf")
        self.hold_until = float("-inf")
        self.predictions = deque(maxlen=5)
        self.status = "Motion: perform J/Z, or press M to record one gesture"
        self.accepted = False

    def _predict(self, samples, start=None):
        probabilities = self.model.predict_proba([resample_motion(samples, start)])[0]
        index = int(np.argmax(probabilities))
        return str(self.model.classes_[index]), float(probabilities[index])

    def toggle_recording(self, now):
        """Return an accepted letter when a manually bounded gesture ends."""
        if not self.recording:
            self.reset()
            self.recording = True
            self.status = "Motion: RECORDING - perform J/Z, then press M again"
            return None

        self.recording = False
        samples = list(self.samples)
        self.samples.clear()
        self.hold_until = now + 3.0
        if len(samples) < SEQUENCE_LENGTH:
            self.status = "Motion: too short - press M and record a longer gesture"
            return None

        label, confidence = self._predict(samples)
        self.accepted = confidence >= self.confidence_threshold
        decision = "accepted" if self.accepted else "below threshold"
        self.status = f"Recorded: {label} {confidence:.0%} - {decision} ({self.confidence_threshold:.0%} needed)"
        print(self.status, flush=True)
        return label if self.accepted else None

    def observe(self, features, now):
        """Keep history independent of static predictions; return accepted motion."""
        if self.last_seen is not None and now - self.last_seen > 0.3:
            self.samples.clear()
            self.predictions.clear()
            self.last_seen = None
            if self.recording:
                self.recording = False
                self.status = "Motion: recording cancelled - hand tracking lost; press M to retry"
                self.hold_until = now + 3.0
            elif now >= self.hold_until:
                self.status = "Motion: hand tracking lost - waiting for a complete gesture"
            self.accepted = False

        if features is None:
            return None
        self.last_seen = now
        if now < self.hold_until:
            return None
        self.samples.append((now, features))

        if self.recording:
            if now - self.samples[0][0] > 10:
                self.recording = False
                self.samples.clear()
                self.status = "Motion: recording timed out - press M to retry"
                self.hold_until = now + 3.0
            return None

        cutoff = now - self.window_seconds
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        if len(self.samples) < SEQUENCE_LENGTH or self.samples[0][0] > cutoff:
            if not self.predictions:
                elapsed = now - self.samples[0][0]
                self.status = f"Motion: collecting {elapsed:.1f}/{self.window_seconds:.1f}s - M records a full gesture"
            return None
        if now - self.last_check < 0.15:
            return None
        self.last_check = now
        label, confidence = self._predict(list(self.samples), cutoff)
        self.accepted = False
        if confidence < self.confidence_threshold:
            self.predictions.clear()
            self.status = f"Motion: {label} {confidence:.0%} - below {self.confidence_threshold:.0%}; try M"
            return None

        self.predictions.append(label)
        stable = len(self.predictions) == 5 and len(set(self.predictions)) == 1
        self.status = f"Motion: {label} {confidence:.0%} - {'accepted' if stable else 'confirming'}"
        if stable:
            self.accepted = True
            self.predictions.clear()
            self.samples.clear()
            self.hold_until = now + 1.0
            return label
        return None
