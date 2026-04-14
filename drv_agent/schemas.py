from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class HallucinationLevel(str, Enum):
    PERCEPTIVE = "perceptive"
    TEMPORAL = "temporal"
    COGNITIVE = "cognitive"


class TaskFormat(str, Enum):
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    CAPTION = "caption"


@dataclass(slots=True)
class QAInput:
    question: str
    options: list[str] = field(default_factory=list)
    task_format: TaskFormat = TaskFormat.MULTIPLE_CHOICE

    def render(self) -> str:
        if not self.options:
            return self.question
        return f"{self.question}\nOptions:\n" + "\n".join(self.options)


@dataclass(slots=True)
class DrVRequest:
    video_path: str
    qa: QAInput
    lvm_answer: str
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityBundle:
    objects: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HallucinationClassification:
    level: HallucinationLevel
    entities: EntityBundle
    raw_response: dict[str, Any] | None = None


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    def intersect(self, other: "BoundingBox") -> "BoundingBox | None":
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


@dataclass(slots=True)
class ObjectObservation:
    timestamp: float
    bbox: BoundingBox
    confidence: float
    source: str


@dataclass(slots=True)
class ObjectEvidence:
    object_name: str
    by_source: dict[str, list[ObjectObservation]] = field(default_factory=dict)
    consensus: list[ObjectObservation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TemporalInterval:
    start: float
    end: float
    confidence: float
    source: str


@dataclass(slots=True)
class EventEvidence:
    event_name: str
    by_source: dict[str, list[TemporalInterval]] = field(default_factory=dict)
    consensus: list[TemporalInterval] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClaimEvidence:
    claim: str
    time_interval: TemporalInterval | None = None
    captions_by_source: dict[str, str] = field(default_factory=dict)
    consensus_caption: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceBundle:
    perceptive: dict[str, ObjectEvidence] = field(default_factory=dict)
    temporal: dict[str, EventEvidence] = field(default_factory=dict)
    cognitive: dict[str, ClaimEvidence] = field(default_factory=dict)


@dataclass(slots=True)
class HallucinationAssessment:
    has_hallucination: bool
    error_points: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: dict[str, Any] | None = None


@dataclass(slots=True)
class StructuredFeedback:
    analysis: str
    recommendations: str
    raw_response: dict[str, Any] | None = None


@dataclass(slots=True)
class DrVReport:
    classification: HallucinationClassification
    evidence: EvidenceBundle
    assessment: HallucinationAssessment
    feedback: StructuredFeedback
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
