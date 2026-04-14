from __future__ import annotations

from dataclasses import dataclass

from ..runtime import AdapterUnavailableError, ensure_sys_path, import_or_raise, resolve_path
from ..schemas import BoundingBox, ObjectObservation
from ..video import FrameBatch


@dataclass(slots=True)
class GroundedSam2Config:
    workspace_root: str
    repo_root: str = "Grounded-SAM-2"
    dino_config: str = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_checkpoint: str = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    sam2_config: str = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.25


@dataclass(slots=True)
class YOLOWorldConfig:
    workspace_root: str
    weight_path: str
    device: str = "cuda"
    confidence_threshold: float = 0.1
    image_size: int = 640


class GroundedSam2ObjectGrounder:
    name = "grounded_sam2"

    def __init__(self, config: GroundedSam2Config):
        self.config = config
        self._grounding_model = None
        self._predictor = None

    def _ensure_loaded(self) -> None:
        if self._grounding_model is not None and self._predictor is not None:
            return

        repo_root = resolve_path(self.config.workspace_root, self.config.repo_root)
        if repo_root is None or not repo_root.exists():
            raise AdapterUnavailableError(f"Grounded-SAM-2 repo not found at: {repo_root}")
        ensure_sys_path(repo_root)
        ensure_sys_path(repo_root / "grounding_dino")

        torch = import_or_raise("torch", "Install torch and torchvision for Grounded-SAM-2.")
        box_convert = import_or_raise(
            "torchvision.ops",
            "Install torchvision for Grounded-SAM-2 box conversion.",
        ).box_convert
        build_sam2 = import_or_raise("sam2.build_sam", "Install Grounded-SAM-2 in editable mode.").build_sam2
        predictor_cls = import_or_raise(
            "sam2.sam2_image_predictor",
            "Install Grounded-SAM-2 in editable mode.",
        ).SAM2ImagePredictor
        inference = import_or_raise(
            "grounding_dino.groundingdino.util.inference",
            "Install Grounded-SAM-2 dependencies and GroundingDINO extras.",
        )

        sam2_model = build_sam2(
            str(repo_root / self.config.sam2_config),
            str(repo_root / self.config.sam2_checkpoint),
            device=self.config.device,
        )
        self._predictor = predictor_cls(sam2_model)
        self._grounding_model = inference.load_model(
            model_config_path=str(repo_root / self.config.dino_config),
            model_checkpoint_path=str(repo_root / self.config.dino_checkpoint),
            device=self.config.device,
        )
        self._torch = torch
        self._box_convert = box_convert
        self._predict = inference.predict

    def detect(self, frame_batch: FrameBatch, object_name: str) -> list[ObjectObservation]:
        self._ensure_loaded()
        prompt = object_name.lower().strip().rstrip(".") + "."
        observations: list[ObjectObservation] = []

        for frame, timestamp in zip(frame_batch.frames, frame_batch.timestamps):
            boxes, confidences, _ = self._predict(
                model=self._grounding_model,
                image=frame,
                caption=prompt,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                device=self.config.device,
            )
            if len(boxes) == 0:
                continue

            h, w, _ = frame.shape
            scaled_boxes = boxes * self._torch.tensor([w, h, w, h], device=boxes.device)
            xyxy = self._box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            best_idx = int(confidences.argmax().item())
            input_box = xyxy[best_idx : best_idx + 1]

            self._predictor.set_image(frame)
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            mask = masks[0].astype(bool)
            if mask.any():
                ys, xs = mask.nonzero()
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                bbox = BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
            else:
                x1, y1, x2, y2 = [float(value) for value in input_box[0]]
                bbox = BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

            observations.append(
                ObjectObservation(
                    timestamp=float(timestamp),
                    bbox=bbox,
                    confidence=float((confidences[best_idx].item() + float(scores[0])) / 2.0),
                    source=self.name,
                )
            )
        return observations


class UltralyticsYoloWorldObjectGrounder:
    name = "yolo_world"

    def __init__(self, config: YOLOWorldConfig):
        self.config = config
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        ultralytics = import_or_raise(
            "ultralytics",
            "Install ultralytics and provide a YOLO-World checkpoint path.",
        )
        weight_path = resolve_path(self.config.workspace_root, self.config.weight_path)
        if weight_path is None or not weight_path.exists():
            raise AdapterUnavailableError(f"YOLO-World checkpoint not found at: {weight_path}")
        self._model = ultralytics.YOLO(str(weight_path))

    def detect(self, frame_batch: FrameBatch, object_name: str) -> list[ObjectObservation]:
        self._ensure_loaded()
        observations: list[ObjectObservation] = []
        try:
            self._model.set_classes([object_name])
        except Exception:
            pass

        for frame, timestamp in zip(frame_batch.frames, frame_batch.timestamps):
            result = self._model.predict(
                frame,
                conf=self.config.confidence_threshold,
                imgsz=self.config.image_size,
                verbose=False,
                device=self.config.device,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            box = result.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [float(value) for value in box]
            observations.append(
                ObjectObservation(
                    timestamp=float(timestamp),
                    bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    confidence=float(result.boxes.conf[0].item()),
                    source=self.name,
                )
            )
        return observations
