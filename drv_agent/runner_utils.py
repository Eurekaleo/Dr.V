from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VideoProbe:
    fps: float
    total_frames: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.total_frames / self.fps


@dataclass(slots=True)
class SampledVideoSegment:
    frames: list[Any]
    timestamps: list[float]
    frame_indices: list[int]
    fps: float
    width: int
    height: int
    total_frames: int


def parse_claim_payload(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {"claim": ""}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"claim": raw}
    if isinstance(payload, dict):
        return payload
    return {"claim": payload}


def select_evenly_spaced_indices(start_index: int, end_index: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if start_index > end_index:
        raise ValueError("start_index must not exceed end_index")

    if start_index == end_index:
        return [start_index]

    span = end_index - start_index
    if count == 1:
        return [start_index + span // 2]

    raw_indices = [
        start_index + round((span * step) / (count - 1))
        for step in range(count)
    ]
    deduped: list[int] = []
    for index in raw_indices:
        clamped = min(max(index, start_index), end_index)
        if not deduped or deduped[-1] != clamped:
            deduped.append(clamped)
    return deduped


def probe_video(video_path: str | Path) -> VideoProbe:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video probing. Install opencv-python.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    if total_frames <= 0:
        raise ValueError(f"Video appears empty: {video_path}")

    return VideoProbe(
        fps=float(fps),
        total_frames=total_frames,
        width=width,
        height=height,
    )


def normalize_interval(
    *,
    start_seconds: float | None,
    end_seconds: float | None,
    video_probe: VideoProbe,
) -> tuple[float, float]:
    duration = video_probe.duration_seconds
    if duration <= 0:
        return 0.0, 0.0

    start = 0.0 if start_seconds is None else max(0.0, float(start_seconds))
    end = duration if end_seconds is None else min(duration, float(end_seconds))

    if start == 0.0 and end == 0.0:
        return 0.0, duration
    if end <= start:
        epsilon = max(1.0 / max(video_probe.fps, 1.0), 0.25)
        end = min(duration, start + epsilon)
    return start, end


def sample_video_interval(
    video_path: str | Path,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    num_segments: int = 8,
) -> SampledVideoSegment:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV and Pillow are required for caption runners. Install opencv-python and pillow."
        ) from exc

    probe = probe_video(video_path)
    start_seconds, end_seconds = normalize_interval(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        video_probe=probe,
    )
    start_index = min(int(round(start_seconds * probe.fps)), probe.total_frames - 1)
    end_index = min(int(round(end_seconds * probe.fps)), probe.total_frames - 1)
    indices = select_evenly_spaced_indices(start_index, max(start_index, end_index), num_segments)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames = []
    timestamps = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        timestamps.append(round(frame_index / probe.fps, 3))
    capture.release()

    if not frames:
        raise ValueError(f"Failed to sample frames from video: {video_path}")

    return SampledVideoSegment(
        frames=frames,
        timestamps=timestamps,
        frame_indices=indices[: len(frames)],
        fps=probe.fps,
        width=probe.width,
        height=probe.height,
        total_frames=probe.total_frames,
    )


def build_dense_caption_prompt(
    *,
    claim: str,
    event_name: str,
    timestamps: list[float],
) -> str:
    claim = claim.strip()
    event_name = event_name.strip()
    frames_note = ""
    if timestamps:
        frames_note = (
            "These frames are ordered snapshots from the same temporal segment at timestamps: "
            + ", ".join(f"{timestamp:.2f}s" for timestamp in timestamps)
            + ". "
        )

    instructions = [
        frames_note,
        "Write one dense factual caption for this segment.",
        "Focus on visible actions, object states, temporal order, and the exact evidence that supports or contradicts the claim.",
        "Do not speculate about off-screen intent or unseen events.",
        "Return only the caption text.",
    ]
    if event_name:
        instructions.insert(1, f"Target event: {event_name}.")
    if claim:
        instructions.insert(1, f"Claim to verify: {claim}.")
    return " ".join(part for part in instructions if part)
