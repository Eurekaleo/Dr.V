from typing import Dict, List, Tuple

import cv2
import numpy as np


class VideoPreprocessor:
    def __init__(self, video_path: str, frame_interval: int = 1):
        """
        Initialize video preprocessor to extract frames and map timestamps.
        """
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frames: List[np.ndarray] = []
        self.frame_timestamps: List[float] = []
        self.fps: float = 0

    def process(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Execute preprocessing: extract frames and compute timestamps.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to read video: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                timestamp = frame_idx / self.fps
                self.frame_timestamps.append(round(timestamp, 2))
            frame_idx += 1

        cap.release()
        if len(self.frames) == 0:
            raise ValueError("No valid frames extracted after preprocessing")
        return self.frames, self.frame_timestamps

    def get_timestamp_by_frame_idx(self, frame_idx: int) -> float:
        """
        Get timestamp corresponding to a specific frame index.
        """
        if frame_idx < 0 or frame_idx >= len(self.frame_timestamps):
            raise IndexError("Frame index out of valid range")
        return self.frame_timestamps[frame_idx]
