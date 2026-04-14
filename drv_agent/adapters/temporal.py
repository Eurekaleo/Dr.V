from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

from ..runtime import AdapterUnavailableError, parse_json_like, resolve_path, run_command_template
from ..schemas import DrVRequest, EntityBundle, EvidenceBundle, TemporalInterval


@dataclass(slots=True)
class GroundedVideoLLMConfig:
    workspace_root: str
    repo_root: str = "Grounded-Video-LLM"
    python_bin: str = "python3"
    script_path: str = "inference.py"
    device: str = "cuda:0"
    llm: str = "phi3.5"
    stage: str = "sft"
    attn_implementation: str = "eager"
    num_frames: int = 96
    num_segs: int = 12
    num_temporal_tokens: int = 300
    max_new_tokens: int = 256
    config_path: str = ""
    tokenizer_path: str = ""
    pretrained_video_path: str = ""
    pretrained_vision_proj_llm_path: str = ""
    ckpt_path: str = ""


@dataclass(slots=True)
class CommandTemporalConfig:
    workspace_root: str
    name: str
    command: str
    working_dir: str = "."
    confidence: float = 0.5


class GroundedVideoLLMTemporalGrounder:
    name = "grounded_videollm"

    def __init__(self, config: GroundedVideoLLMConfig):
        self.config = config

    def ground(
        self,
        request: DrVRequest,
        event_name: str,
        entities: EntityBundle,
        evidence: EvidenceBundle,
    ) -> list[TemporalInterval]:
        repo_root = resolve_path(self.config.workspace_root, self.config.repo_root)
        if repo_root is None or not repo_root.exists():
            raise AdapterUnavailableError(f"Grounded-Video-LLM repo not found at: {repo_root}")

        script_path = resolve_path(repo_root, self.config.script_path)
        if script_path is None or not script_path.exists():
            raise AdapterUnavailableError(f"Grounded-Video-LLM inference script not found at: {script_path}")

        prompt = (
            f"Give you a textual query: '{event_name}'. "
            "When does the described content occur in the video? Please return the start and end timestamps."
        )
        command = [
            self.config.python_bin,
            str(script_path),
            "--video_path",
            request.video_path,
            "--prompt_grounding",
            prompt,
            "--device",
            self.config.device,
            "--llm",
            self.config.llm,
            "--stage",
            self.config.stage,
            "--attn_implementation",
            self.config.attn_implementation,
            "--num_frames",
            str(self.config.num_frames),
            "--num_segs",
            str(self.config.num_segs),
            "--num_temporal_tokens",
            str(self.config.num_temporal_tokens),
            "--max_new_tokens",
            str(self.config.max_new_tokens),
        ]
        optional_args = {
            "--config_path": self.config.config_path,
            "--tokenizer_path": self.config.tokenizer_path,
            "--pretrained_video_path": self.config.pretrained_video_path,
            "--pretrained_vision_proj_llm_path": self.config.pretrained_vision_proj_llm_path,
            "--ckpt_path": self.config.ckpt_path,
        }
        for flag, value in optional_args.items():
            if value:
                command.extend([flag, value])

        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            check=True,
            text=True,
            capture_output=True,
        )
        matches = re.findall(r"(\d+(?:\.\d+)?)\s+seconds", completed.stdout)
        if len(matches) < 2:
            raise AdapterUnavailableError(
                "Failed to parse Grounded-VideoLLM grounding output. Check its checkpoint and prompt response."
            )
        return [
            TemporalInterval(
                start=float(matches[0]),
                end=float(matches[1]),
                confidence=0.5,
                source=self.name,
            )
        ]


class CommandTemporalGrounder:
    def __init__(self, config: CommandTemporalConfig):
        self.config = config
        self.name = config.name

    def ground(
        self,
        request: DrVRequest,
        event_name: str,
        entities: EntityBundle,
        evidence: EvidenceBundle,
    ) -> list[TemporalInterval]:
        completed = run_command_template(
            self.config.command,
            {
                "video_path": request.video_path,
                "event_name": event_name,
                "question": request.qa.question,
                "options_text": "\n".join(request.qa.options),
                "options_json": json.dumps(request.qa.options),
                "lvm_answer": request.lvm_answer,
                "task_id": request.task_id or "",
                "entities_json": json.dumps(
                    {
                        "objects": entities.objects,
                        "events": entities.events,
                        "claims": entities.claims,
                    }
                ),
            },
            cwd=resolve_path(self.config.workspace_root, self.config.working_dir),
        )
        payload = parse_json_like(completed.stdout)
        return [
            TemporalInterval(
                start=float(interval[0]),
                end=float(interval[1]),
                confidence=float(payload.get("confidence", self.config.confidence)),
                source=self.name,
            )
            for interval in payload.get("intervals", [])
        ]
