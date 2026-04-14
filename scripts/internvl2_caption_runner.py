#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drv_agent.runner_utils import build_dense_caption_prompt, parse_claim_payload, sample_video_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL2 dense caption runner for a localized video segment.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--start", type=float, default=0.0, help="Segment start time in seconds.")
    parser.add_argument("--end", type=float, default=0.0, help="Segment end time in seconds.")
    parser.add_argument("--claim", default="", help="Claim JSON or raw claim text.")
    parser.add_argument("--event", default="", help="Optional event description.")
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face id for InternVL2.")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--num-segments", type=int, default=8, help="Number of frames sampled from the interval.")
    parser.add_argument("--input-size", type=int, default=448, help="InternVL image size.")
    parser.add_argument("--max-num", type=int, default=1, help="Maximum tiles per frame.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature when do_sample is enabled.")
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic decoding.")
    parser.add_argument("--use-flash-attn", action="store_true", help="Enable InternVL flash attention loading path.")
    return parser.parse_args()


def resolve_dtype(name: str, device: str):
    import torch

    if name == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios, width: int, height: int, image_size: int):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, *, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda item: item[0] * item[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_image = image.resize((target_width, target_height))
    processed_images = []
    for block_index in range(blocks):
        box = (
            (block_index % (target_width // image_size)) * image_size,
            (block_index // (target_width // image_size)) * image_size,
            ((block_index % (target_width // image_size)) + 1) * image_size,
            ((block_index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_image.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_interval_as_internvl_inputs(args: argparse.Namespace):
    import torch

    sampled = sample_video_interval(
        args.video,
        start_seconds=args.start,
        end_seconds=args.end,
        num_segments=args.num_segments,
    )
    transform = build_transform(args.input_size)
    pixel_values_list = []
    num_patches_list = []
    for frame in sampled.frames:
        tiles = dynamic_preprocess(frame, image_size=args.input_size, max_num=args.max_num, use_thumbnail=True)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
    return sampled, torch.cat(pixel_values_list, dim=0), num_patches_list


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = resolve_dtype(args.dtype, args.device)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.use_flash_attn and args.device.startswith("cuda"):
        load_kwargs["use_flash_attn"] = True
    model = AutoModel.from_pretrained(args.model_path, **load_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if args.device != "auto":
        model = model.to(torch.device(args.device))
    return model, tokenizer, dtype


def infer(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    claim_payload = parse_claim_payload(args.claim)
    claim = str(claim_payload.get("claim", "")).strip()
    sampled, pixel_values, num_patches_list = load_interval_as_internvl_inputs(args)
    model, tokenizer, dtype = load_model(args)
    if args.device != "auto":
        pixel_values = pixel_values.to(dtype=dtype, device=torch.device(args.device))
    else:
        pixel_values = pixel_values.to(dtype=dtype)

    frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    question = frame_prefix + build_dense_caption_prompt(
        claim=claim,
        event_name=args.event,
        timestamps=sampled.timestamps,
    )
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_config["temperature"] = args.temperature

    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
    )
    return {
        "caption": response.strip(),
        "metadata": {
            "model_path": args.model_path,
            "num_frames": len(sampled.frames),
            "timestamps": sampled.timestamps,
            "frame_indices": sampled.frame_indices,
        },
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(infer(args), ensure_ascii=True))


if __name__ == "__main__":
    main()
