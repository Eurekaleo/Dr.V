from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_agent
from .schemas import DrVRequest, QAInput, TaskFormat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dr.V-Agent pipeline.")
    parser.add_argument("command", choices=["run"], help="Command to execute.")
    parser.add_argument("--config", required=True, help="Path to the TOML config.")
    parser.add_argument("--input", required=True, help="Path to the JSON request file.")
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    agent = load_agent(args.config)
    request = _load_request(args.input)
    report = agent.run(request)
    rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


def _load_request(path: str) -> DrVRequest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DrVRequest(
        video_path=payload["video_path"],
        qa=QAInput(
            question=payload["question"],
            options=payload.get("options", []),
            task_format=TaskFormat(payload.get("task_format", "multiple_choice")),
        ),
        lvm_answer=payload["lvm_answer"],
        task_id=payload.get("task_id"),
        metadata=payload.get("metadata", {}),
    )


if __name__ == "__main__":
    main()
