from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FrameBatch:
    frames: list[Any] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    fps: float = 0.0
    frame_indices: list[int] = field(default_factory=list)

    def slice(self, start: float, end: float) -> "FrameBatch":
        frames = []
        timestamps = []
        indices = []
        for frame, timestamp, frame_index in zip(self.frames, self.timestamps, self.frame_indices):
            if start <= timestamp <= end:
                frames.append(frame)
                timestamps.append(timestamp)
                indices.append(frame_index)
        return FrameBatch(frames=frames, timestamps=timestamps, fps=self.fps, frame_indices=indices)


class VideoFrameSampler:
    def __init__(self, frame_interval: int = 8, max_frames: int = 128):
        self.frame_interval = max(1, frame_interval)
        self.max_frames = max_frames

    def sample(self, video_path: str) -> FrameBatch:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for real video sampling. Install opencv-python.") from exc

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        frames = []
        timestamps = []
        frame_indices = []
        frame_index = 0
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(round(frame_index / fps, 2))
                frame_indices.append(frame_index)
                if len(frames) >= self.max_frames:
                    break
            frame_index += 1
        capture.release()

        if not frames:
            raise ValueError(f"No frames were sampled from video: {video_path}")
        return FrameBatch(frames=frames, timestamps=timestamps, fps=fps, frame_indices=frame_indices)
