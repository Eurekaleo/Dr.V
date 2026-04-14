from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from .adapters.cognitive import CommandCaptioner, CommandCaptionerConfig
from .adapters.llm import (
    OpenAIChatFeedbackGenerator,
    OpenAIChatHallucinationClassifier,
    OpenAIChatReasoner,
    OpenAICompatibleConfig,
)
from .adapters.mock import (
    MockCaptioner,
    MockFeedbackGenerator,
    MockFrameSampler,
    MockHallucinationClassifier,
    MockObjectGrounder,
    MockReasoner,
    MockTemporalGrounder,
)
from .adapters.perceptive import GroundedSam2Config, GroundedSam2ObjectGrounder, UltralyticsYoloWorldObjectGrounder, YOLOWorldConfig
from .adapters.temporal import CommandTemporalConfig, CommandTemporalGrounder, GroundedVideoLLMConfig, GroundedVideoLLMTemporalGrounder
from .pipeline import DrVAgent
from .video import VideoFrameSampler


@dataclass(slots=True)
class RunSettings:
    workspace_root: str = "."
    mode: str = "mock"
    device: str = "cpu"
    frame_interval: int = 8
    max_frames: int = 128
    strict_cross_validation: bool = False


@dataclass(slots=True)
class CommandSettings:
    enabled: bool = False
    name: str = ""
    command: str = ""
    working_dir: str = "."


@dataclass(slots=True)
class LLMSettings:
    backend: str = "mock"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    temperature: float = 0.1


@dataclass(slots=True)
class AgentConfig:
    run: RunSettings = field(default_factory=RunSettings)
    classifier: LLMSettings = field(default_factory=LLMSettings)
    reasoner: LLMSettings = field(default_factory=LLMSettings)
    feedback: LLMSettings = field(default_factory=LLMSettings)
    grounded_sam2: dict = field(default_factory=dict)
    yolo_world: dict = field(default_factory=dict)
    grounded_videollm: dict = field(default_factory=dict)
    cgstvg: CommandSettings = field(default_factory=CommandSettings)
    internvl2: CommandSettings = field(default_factory=CommandSettings)
    qwen2vl: CommandSettings = field(default_factory=CommandSettings)


def load_agent(path: str) -> DrVAgent:
    return build_agent(load_config(path))


def load_config(path: str) -> AgentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    run = RunSettings(**raw.get("run", {}))
    run.workspace_root = str((config_path.parent / run.workspace_root).resolve())

    def llm_settings(section: str, default_env: str) -> LLMSettings:
        payload = raw.get(section, {})
        if "api_key_env" not in payload:
            payload["api_key_env"] = default_env
        return LLMSettings(**payload)

    return AgentConfig(
        run=run,
        classifier=llm_settings("classifier", "OPENAI_API_KEY"),
        reasoner=llm_settings("reasoner", "DEEPSEEK_API_KEY"),
        feedback=llm_settings("feedback", "OPENAI_API_KEY"),
        grounded_sam2=raw.get("perceptive", {}).get("grounded_sam2", {}),
        yolo_world=raw.get("perceptive", {}).get("yolo_world", {}),
        grounded_videollm=raw.get("temporal", {}).get("grounded_videollm", {}),
        cgstvg=CommandSettings(**raw.get("temporal", {}).get("cgstvg", {})),
        internvl2=CommandSettings(**raw.get("cognitive", {}).get("internvl2", {})),
        qwen2vl=CommandSettings(**raw.get("cognitive", {}).get("qwen2vl", {})),
    )


def build_agent(config: AgentConfig) -> DrVAgent:
    if config.run.mode == "mock":
        return DrVAgent(
            classifier=MockHallucinationClassifier(),
            reasoner=MockReasoner(),
            feedback_generator=MockFeedbackGenerator(),
            object_grounders=[MockObjectGrounder("grounded_sam2"), MockObjectGrounder("yolo_world")],
            temporal_grounders=[MockTemporalGrounder("cg_stvg"), MockTemporalGrounder("grounded_videollm")],
            captioners=[MockCaptioner("internvl2"), MockCaptioner("qwen2vl")],
            frame_sampler=MockFrameSampler(),
            strict_cross_validation=config.run.strict_cross_validation,
        )

    classifier = _build_llm_component(config.classifier, kind="classifier")
    reasoner = _build_llm_component(config.reasoner, kind="reasoner")
    feedback_generator = _build_llm_component(config.feedback, kind="feedback")

    object_grounders = []
    if config.grounded_sam2.get("enabled"):
        object_grounders.append(
            GroundedSam2ObjectGrounder(
                GroundedSam2Config(
                    workspace_root=config.run.workspace_root,
                    repo_root=config.grounded_sam2.get("repo_root", "Grounded-SAM-2"),
                    dino_config=config.grounded_sam2.get(
                        "dino_config",
                        "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    ),
                    dino_checkpoint=config.grounded_sam2.get(
                        "dino_checkpoint", "gdino_checkpoints/groundingdino_swint_ogc.pth"
                    ),
                    sam2_config=config.grounded_sam2.get(
                        "sam2_config", "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
                    ),
                    sam2_checkpoint=config.grounded_sam2.get(
                        "sam2_checkpoint", "checkpoints/sam2.1_hiera_large.pt"
                    ),
                    device=config.grounded_sam2.get("device", config.run.device),
                    box_threshold=config.grounded_sam2.get("box_threshold", 0.35),
                    text_threshold=config.grounded_sam2.get("text_threshold", 0.25),
                )
            )
        )
    if config.yolo_world.get("enabled"):
        object_grounders.append(
            UltralyticsYoloWorldObjectGrounder(
                YOLOWorldConfig(
                    workspace_root=config.run.workspace_root,
                    weight_path=config.yolo_world.get("weight_path", ""),
                    device=config.yolo_world.get("device", config.run.device),
                    confidence_threshold=config.yolo_world.get("confidence_threshold", 0.1),
                    image_size=config.yolo_world.get("image_size", 640),
                )
            )
        )

    temporal_grounders = []
    if config.cgstvg.enabled and config.cgstvg.command:
        temporal_grounders.append(
            CommandTemporalGrounder(
                CommandTemporalConfig(
                    workspace_root=config.run.workspace_root,
                    name=config.cgstvg.name or "cg_stvg",
                    command=config.cgstvg.command,
                    working_dir=config.cgstvg.working_dir,
                )
            )
        )
    if config.grounded_videollm.get("enabled"):
        temporal_grounders.append(
            GroundedVideoLLMTemporalGrounder(
                GroundedVideoLLMConfig(
                    workspace_root=config.run.workspace_root,
                    repo_root=config.grounded_videollm.get("repo_root", "Grounded-Video-LLM"),
                    python_bin=config.grounded_videollm.get("python_bin", "python3"),
                    script_path=config.grounded_videollm.get("script_path", "inference.py"),
                    device=config.grounded_videollm.get("device", config.run.device),
                    llm=config.grounded_videollm.get("llm", "phi3.5"),
                    stage=config.grounded_videollm.get("stage", "sft"),
                    attn_implementation=config.grounded_videollm.get("attn_implementation", "eager"),
                    num_frames=config.grounded_videollm.get("num_frames", 96),
                    num_segs=config.grounded_videollm.get("num_segs", 12),
                    num_temporal_tokens=config.grounded_videollm.get("num_temporal_tokens", 300),
                    max_new_tokens=config.grounded_videollm.get("max_new_tokens", 256),
                    config_path=config.grounded_videollm.get("config_path", ""),
                    tokenizer_path=config.grounded_videollm.get("tokenizer_path", ""),
                    pretrained_video_path=config.grounded_videollm.get("pretrained_video_path", ""),
                    pretrained_vision_proj_llm_path=config.grounded_videollm.get(
                        "pretrained_vision_proj_llm_path", ""
                    ),
                    ckpt_path=config.grounded_videollm.get("ckpt_path", ""),
                )
            )
        )

    captioners = []
    for command_settings in (config.internvl2, config.qwen2vl):
        if command_settings.enabled and command_settings.command:
            captioners.append(
                CommandCaptioner(
                    CommandCaptionerConfig(
                        workspace_root=config.run.workspace_root,
                        name=command_settings.name or "captioner",
                        command=command_settings.command,
                        working_dir=command_settings.working_dir,
                    )
                )
            )

    return DrVAgent(
        classifier=classifier,
        reasoner=reasoner,
        feedback_generator=feedback_generator,
        object_grounders=object_grounders,
        temporal_grounders=temporal_grounders,
        captioners=captioners,
        frame_sampler=VideoFrameSampler(
            frame_interval=config.run.frame_interval,
            max_frames=config.run.max_frames,
        ),
        strict_cross_validation=config.run.strict_cross_validation,
    )


def _build_llm_component(settings: LLMSettings, *, kind: str):
    if settings.backend == "mock":
        if kind == "classifier":
            return MockHallucinationClassifier()
        if kind == "reasoner":
            return MockReasoner()
        return MockFeedbackGenerator()

    config = OpenAICompatibleConfig(
        model=settings.model,
        api_key_env=settings.api_key_env,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )
    if kind == "classifier":
        return OpenAIChatHallucinationClassifier(config)
    if kind == "reasoner":
        return OpenAIChatReasoner(config)
    return OpenAIChatFeedbackGenerator(config)
