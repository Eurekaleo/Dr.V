from __future__ import annotations

import json
from dataclasses import dataclass

from ..runtime import parse_json_like, resolve_path, run_command_template
from ..schemas import DrVRequest, EvidenceBundle, TemporalInterval
from ..video import FrameBatch


@dataclass(slots=True)
class CommandCaptionerConfig:
    workspace_root: str
    name: str
    command: str
    working_dir: str = "."


class CommandCaptioner:
    def __init__(self, config: CommandCaptionerConfig):
        self.config = config
        self.name = config.name

    def caption(
        self,
        request: DrVRequest,
        claim: str,
        event: str | None,
        interval: TemporalInterval | None,
        frame_batch: FrameBatch | None,
        evidence: EvidenceBundle,
    ) -> str:
        start = interval.start if interval else 0.0
        end = interval.end if interval else 0.0
        completed = run_command_template(
            self.config.command,
            {
                "video_path": request.video_path,
                "claim": claim,
                "event_name": event or "",
                "start": start,
                "end": end,
                "question": request.qa.question,
                "options_text": "\n".join(request.qa.options),
                "options_json": json.dumps(request.qa.options),
                "lvm_answer": request.lvm_answer,
                "task_id": request.task_id or "",
                "claim_json": json.dumps({"claim": claim}),
            },
            cwd=resolve_path(self.config.workspace_root, self.config.working_dir),
        )
        try:
            payload = parse_json_like(completed.stdout)
            return str(payload["caption"])
        except Exception:
            return completed.stdout.strip()
