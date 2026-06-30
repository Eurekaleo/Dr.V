# Dr.V-Bench 

<p align="center">
  <strong><a href="https://link.springer.com/article/10.1007/s11263-026-02831-1">Dr.V: A Hierarchical Perception-Temporal-Cognition Framework to Diagnose Video Hallucination by Fine-Grained Spatial-Temporal Grounding</a></strong>
</p>

<p align="center">
  <strong>International Journal of Computer Vision (IJCV), 2026</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Luo%2C+M"><strong>Meng Luo</strong></a><sup>1</sup>,
  <a href="https://sqwu.top/"><strong>Shengqiong Wu</strong></a><sup>1</sup>,
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Jing%2C+L"><strong>Liqiang Jing</strong></a><sup>2</sup>,
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Ju%2C+T"><strong>Tianjie Ju</strong></a><sup>1</sup>,
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Zheng%2C+L"><strong>Li Zheng</strong></a><sup>3</sup>,
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Lai%2C+J"><strong>Jinxiang Lai</strong></a><sup>4</sup>,
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Wu%2C+T"><strong>Tianlong Wu</strong></a><sup>1</sup>,
  <a href="https://xinyadu.github.io/"><strong>Xinya Du</strong></a><sup>2</sup>,
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Li%2C+J"><strong>Jian Li</strong></a><sup>5</sup>,
  <a href="https://arxiv.org/search/cs?searchtype=author&amp;query=Yan%2C+S"><strong>Siyuan Yan</strong></a><sup>6</sup>,
  <a href="https://www.cs.rochester.edu/u/jluo/"><strong>Jiebo Luo</strong></a><sup>7</sup>,
  <a href="https://sites.cs.ucsb.edu/~william/"><strong>William Yang Wang</strong></a><sup>8</sup>,
  <a href="http://haofei.vip/"><strong>Hao Fei</strong></a><sup>1</sup>,
  <a href="https://www.comp.nus.edu.sg/~leeml/"><strong>Mong-Li Lee</strong></a><sup>1</sup>,
  <a href="https://www.comp.nus.edu.sg/~whsu/"><strong>Wynne Hsu</strong></a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>NUS &nbsp;
  <sup>2</sup>UTD &nbsp;
  <sup>3</sup>WHU &nbsp;
  <sup>4</sup>HKUST &nbsp;
  <sup>5</sup>NJU &nbsp;
  <sup>6</sup>Monash &nbsp;
  <sup>7</sup>UR &nbsp;
  <sup>8</sup>UCSB
</p>

<p align="center">
  <a href="https://link.springer.com/article/10.1007/s11263-026-02831-1"><img src="https://img.shields.io/badge/Paper-Springer-blue" alt="Paper"></a>
  <a href="https://doi.org/10.1007/s11263-026-02831-1"><img src="https://img.shields.io/badge/DOI-10.1007%2Fs11263--026--02831--1-blue" alt="DOI"></a>
  <a href="https://arxiv.org/abs/2509.11866"><img src="https://img.shields.io/badge/arXiv-2509.11866-orange" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2509.11866"><img src="https://img.shields.io/badge/PDF-arXiv-red" alt="PDF"></a>
  <a href="https://huggingface.co/datasets/Eureka-Leo/Dr.V-Bench"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow" alt="Dataset"></a>
</p>

This repository provides an implementation-oriented public release for the paper:

"Dr.V: A Hierarchical Perception-Temporal-Cognition Framework to Diagnose Video Hallucination by Fine-Grained Spatial-Temporal Grounding"

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
