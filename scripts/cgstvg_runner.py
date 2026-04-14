#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-video CG-STVG temporal grounding runner.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--event", required=True, help="Event description to localize.")
    parser.add_argument(
        "--repo-root",
        default="CGSTVG",
        help="Path to the vendored CGSTVG repository root.",
    )
    parser.add_argument(
        "--config-file",
        default="CGSTVG/experiments/hcstvg.yaml",
        help="Path to the CGSTVG config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained CGSTVG checkpoint.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Match the repository's test-time sampling behavior.",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Optional override for cfg.DATASET.NAME. Defaults to the config value.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Optional override for cfg.INPUT.SAMPLE_FPS.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=None,
        help="Optional override for cfg.INPUT.TRAIN_SAMPLE_NUM.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Optional minimum confidence. When unmet, the runner still returns the top interval but annotates it.",
    )
    parser.add_argument(
        "--return-boxes",
        action="store_true",
        help="Include per-frame boxes in the JSON output for debugging.",
    )
    return parser.parse_args()


def bootstrap_repo(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger("cgstvg_runner")


def clone_cfg(config_file: Path, checkpoint: Path, args: argparse.Namespace):
    from config import cfg as base_cfg

    cfg = base_cfg.clone()
    cfg.merge_from_file(str(config_file))
    cfg.defrost()
    cfg.MODEL.WEIGHT = str(checkpoint)
    cfg.MODEL.DEVICE = args.device
    cfg.DATALOADER.NUM_WORKERS = 0
    if args.dataset_name:
        cfg.DATASET.NAME = args.dataset_name
    if args.sample_fps is not None:
        cfg.INPUT.SAMPLE_FPS = args.sample_fps
    if args.sample_frames is not None:
        cfg.INPUT.TRAIN_SAMPLE_NUM = args.sample_frames
    cfg.freeze()
    return cfg


def sample_frame_ids(total_frames: int, fps: float, cfg, *, test_mode: bool) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video must contain at least one frame.")

    if cfg.DATASET.NAME == "HC-STVG":
        input_fps = float(cfg.INPUT.SAMPLE_FPS)
        if test_mode:
            input_fps *= 2.0
        sampling_rate = input_fps / max(fps, 1e-6)
        sample_ids = [0]
        for frame_index in range(total_frames):
            if int(sample_ids[-1] * sampling_rate) < int(frame_index * sampling_rate):
                sample_ids.append(frame_index)
        if sample_ids[-1] != total_frames - 1:
            sample_ids.append(total_frames - 1)
        return sample_ids

    input_frames = int(cfg.INPUT.TRAIN_SAMPLE_NUM)
    if test_mode:
        input_frames *= 2
    if total_frames <= input_frames:
        return list(range(total_frames))

    return sorted(
        {
            min(total_frames - 1, round(((total_frames - 1) * step) / max(input_frames - 1, 1)))
            for step in range(input_frames)
        }
    )


def load_video_clip(video_path: Path, cfg, *, test_mode: bool):
    try:
        import cv2
        import numpy as np
        import torch
        from torchvision.transforms import Resize
    except ImportError as exc:
        raise RuntimeError(
            "CG-STVG runner requires opencv-python, numpy, torch, and torchvision."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_ids = sample_frame_ids(total_frames, fps, cfg, test_mode=test_mode)

    max_rate = 1.4
    resized_width = min(int(cfg.INPUT.RESOLUTION * (width / max(height, 1))), int(cfg.INPUT.RESOLUTION * max_rate))
    resize = Resize((cfg.INPUT.RESOLUTION, resized_width), antialias=True)

    frames = []
    for frame_id in frame_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frames.append(resize(frame_tensor))
    capture.release()

    if not frames:
        raise ValueError(f"Failed to sample any frames from {video_path}")

    clip = torch.stack(frames)
    actioness = np.ones(len(frames), dtype=bool)
    return {
        "clip": clip,
        "frame_ids": frame_ids[: len(frames)],
        "actioness": actioness,
        "fps": fps,
        "height": height,
        "width": width,
        "total_frames": total_frames,
    }


def build_eval_inputs(video_item: dict, event: str, cfg):
    import torch

    from datasets.build import build_transforms
    from utils.bounding_box import BoxList
    from utils.misc import NestedTensor, to_device

    dummy_boxes = torch.tensor(
        [[0.0, 0.0, float(video_item["width"]), float(video_item["height"])] for _ in video_item["frame_ids"]],
        dtype=torch.float32,
    )
    input_dict = {
        "frames": video_item["clip"],
        "boxs": BoxList(dummy_boxes, (video_item["width"], video_item["height"]), "xyxy"),
        "text": event.lower(),
        "actioness": video_item["actioness"],
    }
    transforms = build_transforms(cfg, is_train=False)
    input_dict = transforms(input_dict)

    target = {
        "item_id": Path("video").stem,
        "ori_size": (video_item["height"], video_item["width"]),
        "frame_ids": video_item["frame_ids"],
        "boxs": input_dict["boxs"].bbox.clone(),
        "actioness": torch.from_numpy(video_item["actioness"]).bool(),
        "eval": True,
    }
    nested = NestedTensor.from_tensor_list([input_dict["frames"]])
    device = torch.device(cfg.MODEL.DEVICE)
    nested = nested.to(device)
    targets = to_device([target], device)
    return nested, [event.lower()], targets


def temporal_confidence(pred_sted, durations) -> list[float]:
    import torch
    import torch.nn.functional as F

    batch_size, time_steps, _ = pred_sted.shape
    device = pred_sted.device
    confidence_scores = []
    for batch_index in range(batch_size):
        duration = durations[batch_index]
        probability_map = torch.full((time_steps, time_steps), -1e32, device=device).tril(0)
        probability_map[duration:, :] = -1e32
        probability_map[:, duration:] = -1e32
        probability_map += (
            F.log_softmax(pred_sted[batch_index, :, 0], dim=0).unsqueeze(1)
            + F.log_softmax(pred_sted[batch_index, :, 1], dim=0).unsqueeze(0)
        )
        confidence_scores.append(float(probability_map.max().exp().item()))
    return confidence_scores


def single_clip_inference(cfg, model, postprocessor, videos, texts, targets, device):
    import torch

    durations = videos.durations
    targets[0]["durations"] = durations
    with torch.inference_mode():
        outputs = model(videos, texts, targets)

    batch_size = len(durations)
    time_steps = max(durations)
    batch_img_size = [list(target["ori_size"]) for target in targets]
    orig_target_sizes = [img_size for img_size in batch_img_size for _ in range(time_steps)]
    orig_target_sizes = torch.tensor(orig_target_sizes, device=device)
    frames_ids = [target["frame_ids"] for target in targets]

    pred_boxes, pred_steds = postprocessor(outputs, orig_target_sizes, frames_ids, durations)
    pred_boxes = pred_boxes.view(batch_size, time_steps, 4)
    confidences = temporal_confidence(outputs["pred_sted"], durations)

    bbox_pred = {}
    temporal_pred = {}
    for batch_index in range(batch_size):
        item_id = targets[batch_index]["item_id"]
        bbox_pred[item_id] = {}
        for offset, frame_id in enumerate(frames_ids[batch_index]):
            bbox_pred[item_id][frame_id] = [pred_boxes[batch_index][offset].detach().cpu().tolist()]
        temporal_pred[item_id] = {
            "sted": pred_steds[batch_index],
            "confidence": confidences[batch_index],
        }
    return bbox_pred, temporal_pred


def merge_two_pass_predictions(first_boxes, first_temporal, second_boxes, second_temporal):
    from engine.evaluate import linear_interp

    merged_boxes = {}
    merged_temporal = {}
    for video_id in first_boxes:
        first_boxes[video_id].update(second_boxes.get(video_id, {}))
        merged_boxes[video_id] = linear_interp(first_boxes[video_id])
        merged_temporal[video_id] = {
            "sted": [
                min(first_temporal[video_id]["sted"][0], second_temporal[video_id]["sted"][0]),
                max(first_temporal[video_id]["sted"][1], second_temporal[video_id]["sted"][1]),
            ],
            "confidence": (first_temporal[video_id]["confidence"] + second_temporal[video_id]["confidence"]) / 2.0,
        }
    return merged_boxes, merged_temporal


def run_inference(args: argparse.Namespace) -> dict:
    import torch

    repo_root = Path(args.repo_root).resolve()
    config_file = Path(args.config_file).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    video_path = Path(args.video).resolve()

    if not repo_root.exists():
        raise FileNotFoundError(f"CGSTVG repo root not found: {repo_root}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    bootstrap_repo(repo_root)

    from models import build_model, build_postprocessors
    from utils.checkpoint import VSTGCheckpointer

    logger = build_logger()
    cfg = clone_cfg(config_file, checkpoint, args)
    model, _, _ = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model.eval()

    checkpointer = VSTGCheckpointer(cfg, model, logger=logger, is_train=False)
    checkpointer.load(cfg.MODEL.WEIGHT, with_optim=False)
    postprocessor = build_postprocessors()

    sampled = load_video_clip(video_path, cfg, test_mode=args.test_mode)
    videos, texts, targets = build_eval_inputs(sampled, args.event, cfg)
    targets[0]["item_id"] = video_path.stem

    if len(sampled["frame_ids"]) == 1:
        merged_boxes, merged_temporal = single_clip_inference(cfg, model, postprocessor, videos, texts, targets, device)
    else:
        videos_first = videos.subsample(2, start_idx=0)
        videos_second = videos.subsample(2, start_idx=1)

        target_first = [
            {
                "item_id": targets[0]["item_id"],
                "ori_size": targets[0]["ori_size"],
                "frame_ids": targets[0]["frame_ids"][0::2],
                "boxs": targets[0]["boxs"][0::2].clone(),
                "actioness": targets[0]["actioness"][0::2].clone(),
                "eval": True,
            }
        ]
        target_second = [
            {
                "item_id": targets[0]["item_id"],
                "ori_size": targets[0]["ori_size"],
                "frame_ids": targets[0]["frame_ids"][1::2],
                "boxs": targets[0]["boxs"][1::2].clone(),
                "actioness": targets[0]["actioness"][1::2].clone(),
                "eval": True,
            }
        ]

        if not target_second[0]["frame_ids"]:
            merged_boxes, merged_temporal = single_clip_inference(
                cfg, model, postprocessor, videos_first, texts, target_first, device
            )
        else:
            first_boxes, first_temporal = single_clip_inference(
                cfg, model, postprocessor, videos_first, texts, target_first, device
            )
            second_boxes, second_temporal = single_clip_inference(
                cfg, model, postprocessor, videos_second, texts, target_second, device
            )
            merged_boxes, merged_temporal = merge_two_pass_predictions(
                first_boxes, first_temporal, second_boxes, second_temporal
            )

    temporal_prediction = merged_temporal[video_path.stem]
    start_frame, end_frame = temporal_prediction["sted"]
    confidence = temporal_prediction["confidence"]
    response = {
        "intervals": [[round(start_frame / sampled["fps"], 3), round(end_frame / sampled["fps"], 3)]],
        "confidence": round(confidence, 6),
        "frame_interval": [int(start_frame), int(end_frame)],
        "fps": sampled["fps"],
        "metadata": {
            "dataset_name": cfg.DATASET.NAME,
            "num_input_frames": len(sampled["frame_ids"]),
            "total_frames": sampled["total_frames"],
            "event": args.event,
            "confidence_threshold": args.confidence_threshold,
            "below_threshold": confidence < args.confidence_threshold,
        },
    }
    if args.return_boxes:
        response["boxes_by_frame"] = merged_boxes[video_path.stem]
    return response


def main() -> None:
    args = parse_args()
    payload = run_inference(args)
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
