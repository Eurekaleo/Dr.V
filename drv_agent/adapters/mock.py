from __future__ import annotations

import re

from ..schemas import (
    BoundingBox,
    DrVRequest,
    EntityBundle,
    EvidenceBundle,
    HallucinationAssessment,
    HallucinationClassification,
    HallucinationLevel,
    ObjectObservation,
    StructuredFeedback,
    TemporalInterval,
)
from ..video import FrameBatch


class MockHallucinationClassifier:
    def classify(self, request: DrVRequest) -> HallucinationClassification:
        text = f"{request.qa.question} {request.lvm_answer}".lower()
        if any(keyword in text for keyword in ("why", "because", "reason")):
            level = HallucinationLevel.COGNITIVE
        elif any(keyword in text for keyword in ("before", "after", "then", "when", "while")):
            level = HallucinationLevel.TEMPORAL
        else:
            level = HallucinationLevel.PERCEPTIVE

        objects = _extract_keywords(text, ("baby", "bookshelf", "toy", "man", "car", "woods", "pond"))
        events = _extract_keywords(text, ("walk", "pick", "throw", "put", "cry"))
        claims = [request.lvm_answer] if level == HallucinationLevel.COGNITIVE else []
        return HallucinationClassification(
            level=level,
            entities=EntityBundle(objects=objects, events=events, claims=claims),
            raw_response={"mode": "mock"},
        )


class MockObjectGrounder:
    def __init__(self, name: str):
        self.name = name

    def detect(self, frame_batch: FrameBatch, object_name: str) -> list[ObjectObservation]:
        timestamps = frame_batch.timestamps or [1.0, 2.0]
        observations = []
        for offset, timestamp in enumerate(timestamps[:2]):
            observations.append(
                ObjectObservation(
                    timestamp=float(timestamp),
                    bbox=BoundingBox(x=20.0 + offset * 5, y=30.0, w=120.0, h=160.0),
                    confidence=0.85,
                    source=self.name,
                )
            )
        return observations


class MockTemporalGrounder:
    def __init__(self, name: str):
        self.name = name

    def ground(
        self,
        request: DrVRequest,
        event_name: str,
        entities: EntityBundle,
        evidence: EvidenceBundle,
    ) -> list[TemporalInterval]:
        return [TemporalInterval(start=1.0, end=3.0, confidence=0.8, source=self.name)]


class MockCaptioner:
    def __init__(self, name: str):
        self.name = name

    def caption(
        self,
        request: DrVRequest,
        claim: str,
        event: str | None,
        interval: TemporalInterval | None,
        frame_batch: FrameBatch | None,
        evidence: EvidenceBundle,
    ) -> str:
        if "man" in request.lvm_answer.lower():
            return "The baby noticed something near the bookshelf and walked toward a toy."
        return f"Cause-effect evidence for claim: {claim}"


class MockReasoner:
    def assess(self, request: DrVRequest, evidence: EvidenceBundle) -> HallucinationAssessment:
        hallucination = "play with man" in request.lvm_answer.lower()
        errors = []
        if hallucination:
            errors.append("The answer introduces a man-based cause not supported by the mock evidence.")
        return HallucinationAssessment(
            has_hallucination=hallucination,
            error_points=errors,
            confidence=0.91 if hallucination else 0.74,
            raw_response={"mode": "mock"},
        )


class MockFeedbackGenerator:
    def generate(
        self,
        request: DrVRequest,
        evidence: EvidenceBundle,
        assessment: HallucinationAssessment,
    ) -> StructuredFeedback:
        if assessment.has_hallucination:
            return StructuredFeedback(
                analysis="The evidence indicates the claimed cause is unsupported by the video.",
                recommendations="Re-answer using the grounded object, event, and causal evidence only.",
                raw_response={"mode": "mock"},
            )
        return StructuredFeedback(
            analysis="The answer is consistent with the available evidence.",
            recommendations="No correction is needed.",
            raw_response={"mode": "mock"},
        )


class MockFrameSampler:
    def sample(self, video_path: str) -> FrameBatch:
        return FrameBatch(
            frames=[None, None],
            timestamps=[1.0, 2.0],
            fps=1.0,
            frame_indices=[1, 2],
        )


def _extract_keywords(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    matches = []
    for word in vocabulary:
        if re.search(rf"\b{re.escape(word)}\b", text):
            matches.append(word)
    return matches
