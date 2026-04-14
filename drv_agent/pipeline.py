from __future__ import annotations

from dataclasses import dataclass, field

from .adapters.base import Captioner, FeedbackGenerator, HallucinationClassifier, ObjectGrounder, Reasoner, TemporalGrounder
from .runtime import AdapterUnavailableError
from .schemas import (
    ClaimEvidence,
    DrVReport,
    DrVRequest,
    EvidenceBundle,
    EventEvidence,
    HallucinationLevel,
    ObjectEvidence,
    ObjectObservation,
    TemporalInterval,
)
from .video import VideoFrameSampler


@dataclass(slots=True)
class DrVAgent:
    classifier: HallucinationClassifier
    reasoner: Reasoner
    feedback_generator: FeedbackGenerator
    object_grounders: list[ObjectGrounder] = field(default_factory=list)
    temporal_grounders: list[TemporalGrounder] = field(default_factory=list)
    captioners: list[Captioner] = field(default_factory=list)
    frame_sampler: VideoFrameSampler = field(default_factory=VideoFrameSampler)
    strict_cross_validation: bool = False

    def run(self, request: DrVRequest) -> DrVReport:
        warnings: list[str] = []
        classification = self.classifier.classify(request)
        evidence = EvidenceBundle()

        frame_batch = None
        if classification.entities.objects or classification.level == HallucinationLevel.COGNITIVE:
            try:
                frame_batch = self.frame_sampler.sample(request.video_path)
            except Exception as exc:
                if self.strict_cross_validation:
                    raise
                warnings.append(f"Frame sampling failed: {exc}")

        if classification.entities.objects:
            if not self.object_grounders:
                self._handle_missing_tools("perceptive grounding", warnings)
            elif frame_batch is not None:
                for object_name in classification.entities.objects:
                    evidence.perceptive[object_name] = self._build_object_evidence(frame_batch, object_name, warnings)

        if classification.level in {HallucinationLevel.TEMPORAL, HallucinationLevel.COGNITIVE} and classification.entities.events:
            if not self.temporal_grounders:
                self._handle_missing_tools("temporal grounding", warnings)
            else:
                for event_name in classification.entities.events:
                    evidence.temporal[event_name] = self._build_event_evidence(
                        request=request,
                        event_name=event_name,
                        entities=classification.entities,
                        evidence=evidence,
                        warnings=warnings,
                    )

        if classification.level == HallucinationLevel.COGNITIVE and classification.entities.claims:
            if not self.captioners:
                self._handle_missing_tools("cognitive captioning", warnings)
            else:
                for claim in classification.entities.claims:
                    evidence.cognitive[claim] = self._build_claim_evidence(
                        request=request,
                        claim=claim,
                        evidence=evidence,
                        frame_batch=frame_batch,
                        warnings=warnings,
                    )

        assessment = self.reasoner.assess(request, evidence)
        feedback = self.feedback_generator.generate(request, evidence, assessment)
        return DrVReport(
            classification=classification,
            evidence=evidence,
            assessment=assessment,
            feedback=feedback,
            warnings=warnings,
        )

    def _build_object_evidence(self, frame_batch, object_name: str, warnings: list[str]) -> ObjectEvidence:
        by_source: dict[str, list[ObjectObservation]] = {}
        notes: list[str] = []
        for grounder in self.object_grounders:
            try:
                by_source[grounder.name] = grounder.detect(frame_batch, object_name)
            except AdapterUnavailableError as exc:
                notes.append(str(exc))
        consensus = _cross_validate_observations(by_source)
        if self.strict_cross_validation and len(by_source) < 2:
            raise RuntimeError(f"Perceptive step requires two tools; got {list(by_source)}")
        if len(by_source) < 2:
            warnings.append(f"Perceptive step for '{object_name}' ran with fewer than two tools.")
        return ObjectEvidence(object_name=object_name, by_source=by_source, consensus=consensus, notes=notes)

    def _build_event_evidence(
        self,
        *,
        request: DrVRequest,
        event_name: str,
        entities,
        evidence: EvidenceBundle,
        warnings: list[str],
    ) -> EventEvidence:
        by_source: dict[str, list[TemporalInterval]] = {}
        notes: list[str] = []
        for grounder in self.temporal_grounders:
            try:
                by_source[grounder.name] = grounder.ground(request, event_name, entities, evidence)
            except AdapterUnavailableError as exc:
                notes.append(str(exc))
        consensus = _cross_validate_intervals(by_source)
        if self.strict_cross_validation and len(by_source) < 2:
            raise RuntimeError(f"Temporal step requires two tools; got {list(by_source)}")
        if len(by_source) < 2:
            warnings.append(f"Temporal step for '{event_name}' ran with fewer than two tools.")
        return EventEvidence(event_name=event_name, by_source=by_source, consensus=consensus, notes=notes)

    def _build_claim_evidence(
        self,
        *,
        request: DrVRequest,
        claim: str,
        evidence: EvidenceBundle,
        frame_batch,
        warnings: list[str],
    ) -> ClaimEvidence:
        related_event = _match_event_for_claim(claim, evidence.temporal)
        interval = None
        if related_event and evidence.temporal[related_event].consensus:
            interval = evidence.temporal[related_event].consensus[0]

        captions_by_source: dict[str, str] = {}
        notes: list[str] = []
        clipped_frames = frame_batch.slice(interval.start, interval.end) if frame_batch and interval else frame_batch
        for captioner in self.captioners:
            try:
                captions_by_source[captioner.name] = captioner.caption(
                    request=request,
                    claim=claim,
                    event=related_event,
                    interval=interval,
                    frame_batch=clipped_frames,
                    evidence=evidence,
                )
            except AdapterUnavailableError as exc:
                notes.append(str(exc))
        if self.strict_cross_validation and len(captions_by_source) < 2:
            raise RuntimeError(f"Cognitive step requires two captioners; got {list(captions_by_source)}")
        if len(captions_by_source) < 2:
            warnings.append(f"Cognitive step for '{claim}' ran with fewer than two tools.")
        return ClaimEvidence(
            claim=claim,
            time_interval=interval,
            captions_by_source=captions_by_source,
            consensus_caption=_consensus_caption(captions_by_source),
            notes=notes,
        )

    def _handle_missing_tools(self, label: str, warnings: list[str]) -> None:
        if self.strict_cross_validation:
            raise RuntimeError(f"Missing required tools for {label}.")
        warnings.append(f"Skipping {label} because no tools were configured.")


def _cross_validate_observations(by_source: dict[str, list[ObjectObservation]]) -> list[ObjectObservation]:
    if not by_source:
        return []
    if len(by_source) == 1:
        return next(iter(by_source.values()))

    source_names = list(by_source)
    left = by_source[source_names[0]]
    right = by_source[source_names[1]]
    right_by_timestamp = {round(item.timestamp, 2): item for item in right}
    consensus: list[ObjectObservation] = []
    for item in left:
        partner = right_by_timestamp.get(round(item.timestamp, 2))
        if partner is None:
            continue
        intersection = item.bbox.intersect(partner.bbox)
        if intersection is None:
            continue
        consensus.append(
            ObjectObservation(
                timestamp=round((item.timestamp + partner.timestamp) / 2.0, 2),
                bbox=intersection,
                confidence=(item.confidence + partner.confidence) / 2.0,
                source="intersection",
            )
        )
    return consensus


def _cross_validate_intervals(by_source: dict[str, list[TemporalInterval]]) -> list[TemporalInterval]:
    if not by_source:
        return []
    if len(by_source) == 1:
        return next(iter(by_source.values()))
    source_names = list(by_source)
    left = by_source[source_names[0]]
    right = by_source[source_names[1]]
    if not left or not right:
        return []
    start = max(left[0].start, right[0].start)
    end = min(left[0].end, right[0].end)
    if end <= start:
        return []
    return [
        TemporalInterval(
            start=start,
            end=end,
            confidence=(left[0].confidence + right[0].confidence) / 2.0,
            source="intersection",
        )
    ]


def _match_event_for_claim(claim: str, temporal_evidence: dict[str, EventEvidence]) -> str | None:
    lowered_claim = claim.lower()
    for event_name in temporal_evidence:
        if event_name.lower() in lowered_claim:
            return event_name
    return next(iter(temporal_evidence), None)


def _consensus_caption(captions_by_source: dict[str, str]) -> str | None:
    if not captions_by_source:
        return None
    unique = list(dict.fromkeys(value.strip() for value in captions_by_source.values() if value.strip()))
    if len(unique) == 1:
        return unique[0]
    return " | ".join(unique)
