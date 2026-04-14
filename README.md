# Dr.V-Bench 

This repository provides an implementation-oriented public release for the paper:

"Dr.V: A Hierarchical Perception-Temporal-Cognition Framework to Diagnose Video Hallucination by Fine-grained Spatial-Temporal Grounding"

It includes the Dr.V-Agent diagnosis pipeline as a runnable Python package and retains only the components necessary for reproducing the paper’s workflow.

The dataset is available at:
https://huggingface.co/datasets/Eureka-Leo/Dr.V-Bench


## Method

`Dr.V-Agent` follows the six-stage procedure described in the paper:

1. hallucination type classification
2. perceptive grounding
3. temporal grounding
4. cognitive verification
5. reasoning
6. feedback generation

The tool mapping in this release is:

- perceptive grounding: `Grounded-SAM-2` and `YOLO-World`
- temporal grounding: `CG-STVG` and `Grounded-Video-LLM`
- cognitive verification: `InternVL2` and `Qwen2-VL`
- reasoning: DeepSeek-R1-compatible model
- classification and feedback: GPT-4o-compatible model

## Repository Layout

- `drv_agent/`: orchestration, schemas, config loading, prompts, and adapters
- `scripts/`: concrete runners for `CG-STVG`, `InternVL2`, and `Qwen2-VL`
- `CGSTVG/`: minimal vendored runtime subset for temporal grounding
- `Grounded-SAM-2/`: minimal vendored runtime subset for perceptive grounding
- `Grounded-Video-LLM/`: minimal vendored runtime subset for temporal grounding
- `examples/real_config.example.toml`: real deployment template
- `examples/request.example.json`: request payload example

`YOLO-World` is used through the external runtime expected by `ultralytics.YOLO`; a vendored source copy is not required by this release.

## Setup

Install the project package:

```bash
pip install -e .
```

Then prepare the external runtime dependencies and checkpoints required by the paper tools:

- OpenAI-compatible endpoint for classification and feedback
- DeepSeek-R1-compatible endpoint for reasoning
- Grounded-SAM-2 checkpoints
- YOLO-World checkpoint compatible with `ultralytics`
- CG-STVG checkpoint
- Grounded-Video-LLM checkpoint set
- InternVL2 checkpoint
- Qwen2-VL checkpoint

## Configuration

Use `examples/real_config.example.toml` as the reference configuration.

Important runtime entrypoints:

- `Grounded-SAM-2`: direct Python integration from the vendored runtime
- `CG-STVG`: `scripts/cgstvg_runner.py`
- `Grounded-Video-LLM`: wrapped from `Grounded-Video-LLM/inference.py`
- `InternVL2`: `scripts/internvl2_caption_runner.py`
- `Qwen2-VL`: `scripts/qwen2vl_caption_runner.py`

## Running

Example:

```bash
python3 -m drv_agent.cli run \
  --config examples/real_config.example.toml \
  --input examples/request.example.json
```

The input JSON must provide:

- `video_path`
- `question`
- `options`
- `lvm_answer`

The output report contains:

- `classification`
- `evidence.perceptive`
- `evidence.temporal`
- `evidence.cognitive`
- `assessment`
- `feedback`
- `warnings`

## Notes

- Third-party dependencies included in this repository have been reduced to the minimal subset required for runtime.
- Model weights are intentionally not included.
- Exact reproduction of experiments requires access to the same checkpoints and service endpoints used in the original environment.
