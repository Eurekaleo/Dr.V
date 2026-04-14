from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..prompting import classification_prompt, feedback_prompt, reasoning_prompt
from ..runtime import AdapterUnavailableError, parse_json_like
from ..schemas import (
    DrVRequest,
    EntityBundle,
    EvidenceBundle,
    HallucinationAssessment,
    HallucinationClassification,
    HallucinationLevel,
    StructuredFeedback,
)


@dataclass(slots=True)
class OpenAICompatibleConfig:
    model: str
    api_key_env: str
    base_url: str | None = None
    temperature: float = 0.1


class _OpenAIJSONClient:
    def __init__(self, config: OpenAICompatibleConfig):
        try:
            import os
            from openai import OpenAI
        except ImportError as exc:
            raise AdapterUnavailableError(
                "The openai package is required for GPT-4o / DeepSeek-backed steps. Install openai>=1.0."
            ) from exc

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise AdapterUnavailableError(
                f"Environment variable '{config.api_key_env}' is required for model '{config.model}'."
            )

        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config

    def complete_json(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
        )
        content = response.choices[0].message.content or "{}"
        return parse_json_like(content)


class OpenAIChatHallucinationClassifier:
    def __init__(self, config: OpenAICompatibleConfig):
        self.client = _OpenAIJSONClient(config)

    def classify(self, request: DrVRequest) -> HallucinationClassification:
        payload = self.client.complete_json(classification_prompt(request))
        level = HallucinationLevel(payload["hallucination_level"].lower())
        entities = payload.get("entities", {})
        return HallucinationClassification(
            level=level,
            entities=EntityBundle(
                objects=entities.get("O", []),
                events=entities.get("E", []),
                claims=entities.get("C", []),
            ),
            raw_response=payload,
        )


class OpenAIChatReasoner:
    def __init__(self, config: OpenAICompatibleConfig):
        self.client = _OpenAIJSONClient(config)

    def assess(self, request: DrVRequest, evidence: EvidenceBundle) -> HallucinationAssessment:
        payload = self.client.complete_json(reasoning_prompt(request, evidence))
        return HallucinationAssessment(
            has_hallucination=bool(payload.get("has_hallucination", False)),
            error_points=list(payload.get("error_points", [])),
            confidence=float(payload.get("confidence", 0.0)),
            raw_response=payload,
        )


class OpenAIChatFeedbackGenerator:
    def __init__(self, config: OpenAICompatibleConfig):
        self.client = _OpenAIJSONClient(config)

    def generate(
        self,
        request: DrVRequest,
        evidence: EvidenceBundle,
        assessment: HallucinationAssessment,
    ) -> StructuredFeedback:
        payload = self.client.complete_json(feedback_prompt(request, evidence, assessment))
        feedback = payload.get("feedback", {})
        return StructuredFeedback(
            analysis=feedback.get("A", ""),
            recommendations=feedback.get("R", ""),
            raw_response=payload,
        )
