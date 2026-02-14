"""MediaPipe-based hand bbox detector for HaMeR.

Replaces the ViTPose+detectron2 pipeline with a single lightweight MediaPipe model.
Keeps only the largest detection per hand side and filters by min area fraction.

Download model: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

from __future__ import annotations

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeHandDetector:
    """Detects hand bounding boxes using MediaPipe Hand Landmarker.

    Returns (boxes[N,4], is_right[N]) compatible with HaMeR's input format.
    """

    def __init__(self, model_path: str, min_area_fraction: float = 0.004, padding: float = 0.1):
        self._model_path = model_path
        self.min_area_fraction = min_area_fraction
        self.padding = padding
        self._create_detector()

    def _create_detector(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=4,  # detect extra so we can filter to the best
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def reset(self):
        """Reset internal state for processing a new video."""
        self.detector.close()
        self._create_detector()

    def detect(self, img_rgb: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Detect hands, return (boxes[N,4], is_right[N]) or (None, None)."""
        h, w = img_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not results.hand_landmarks:
            return None, None

        # Keep largest detection per hand side
        best: dict[str, tuple[float, list[int]]] = {}
        for landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            label = handedness[0].category_name
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))

            pad = int(self.padding * max(x1 - x0, y1 - y0))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)

            area = (x1 - x0) * (y1 - y0)
            if label not in best or area > best[label][0]:
                best[label] = (area, [x0, y0, x1, y1])

        # Filter by min area fraction, build output
        frame_area = w * h
        boxes, is_right = [], []
        for label, (area, bbox) in best.items():
            if area / frame_area < self.min_area_fraction:
                continue
            boxes.append(bbox)
            # MediaPipe classifies by visual appearance: "Right" = right hand
            is_right.append(1 if label == "Right" else 0)

        if not boxes:
            return None, None
        return np.array(boxes), np.array(is_right)

    def close(self):
        self.detector.close()
