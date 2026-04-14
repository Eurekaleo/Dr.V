from __future__ import annotations

from typing import Protocol

from ..schemas import (
    DrVRequest,
    EntityBundle,
    EvidenceBundle,
    HallucinationAssessment,
    HallucinationClassification,
    ObjectObservation,
    StructuredFeedback,
    TemporalInterval,
)
from ..video import FrameBatch


class HallucinationClassifier(Protocol):
    def classify(self, request: DrVRequest) -> HallucinationClassification: ...


class ObjectGrounder(Protocol):
    name: str

    def detect(self, frame_batch: FrameBatch, object_name: str) -> list[ObjectObservation]: ...


class TemporalGrounder(Protocol):
    name: str

    def ground(
        self,
        request: DrVRequest,
        event_name: str,
        entities: EntityBundle,
        evidence: EvidenceBundle,
    ) -> list[TemporalInterval]: ...


class Captioner(Protocol):
    name: str

    def caption(
        self,
        request: DrVRequest,
        claim: str,
        event: str | None,
        interval: TemporalInterval | None,
        frame_batch: FrameBatch | None,
        evidence: EvidenceBundle,
    ) -> str: ...


class Reasoner(Protocol):
    def assess(self, request: DrVRequest, evidence: EvidenceBundle) -> HallucinationAssessment: ...


class FeedbackGenerator(Protocol):
    def generate(
        self,
        request: DrVRequest,
        evidence: EvidenceBundle,
        assessment: HallucinationAssessment,
    ) -> StructuredFeedback: ...
