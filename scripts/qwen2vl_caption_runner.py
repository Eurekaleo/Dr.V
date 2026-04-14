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
    parser = argparse.ArgumentParser(description="Qwen2-VL dense caption runner for a localized video segment.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--start", type=float, default=0.0, help="Segment start time in seconds.")
    parser.add_argument("--end", type=float, default=0.0, help="Segment end time in seconds.")
    parser.add_argument("--claim", default="", help="Claim JSON or raw claim text.")
    parser.add_argument("--event", default="", help="Optional event description.")
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face id for Qwen2-VL.")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--num-segments", type=int, default=8, help="Number of frames sampled from the interval.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature when do_sample is enabled.")
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic decoding.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional Qwen2-VL processor min_pixels.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional Qwen2-VL processor max_pixels.")
    parser.add_argument(
        "--attn-implementation",
        default="",
        help="Optional attention implementation, for example flash_attention_2.",
    )
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


def load_model_and_processor(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = resolve_dtype(args.dtype, args.device)
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs).eval()
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    if args.device != "auto":
        model = model.to(torch.device(args.device))
    return model, processor


def infer(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    claim_payload = parse_claim_payload(args.claim)
    claim = str(claim_payload.get("claim", "")).strip()
    sampled = sample_video_interval(
        args.video,
        start_seconds=args.start,
        end_seconds=args.end,
        num_segments=args.num_segments,
    )
    model, processor = load_model_and_processor(args)

    prompt_text = build_dense_caption_prompt(
        claim=claim,
        event_name=args.event,
        timestamps=sampled.timestamps,
    )
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in sampled.frames] + [{"type": "text", "text": prompt_text}],
        }
    ]
    chat_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        add_vision_id=True,
        tokenize=False,
    )
    inputs = processor(
        text=[chat_prompt],
        images=sampled.frames,
        padding=True,
        return_tensors="pt",
    )
    if args.device != "auto":
        inputs = {key: value.to(torch.device(args.device)) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
    output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
    return {
        "caption": output_text,
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
