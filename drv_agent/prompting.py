from __future__ import annotations

import json

from .schemas import DrVRequest, EvidenceBundle, HallucinationAssessment


def classification_prompt(request: DrVRequest) -> str:
    taxonomy = {
        "perceptive": [
            "object recognition",
            "color identification",
            "number estimation",
            "location",
            "static relation",
            "OCR",
        ],
        "temporal": [
            "action recognition",
            "dynamic attribute",
            "dynamic relation",
            "sequence understanding",
        ],
        "cognitive": [
            "factual prediction",
            "counterfactual prediction",
            "context-based explanation",
            "knowledge-based explanation",
        ],
    }
    return f"""
You are implementing Step 1 of Dr.V-Agent from the paper "Dr.V: A Hierarchical Perception-Temporal-Cognition Framework to Diagnose Video Hallucination by Fine-grained Spatial-Temporal Grounding".

Task:
1. Classify the likely hallucination level of the answer as exactly one of: perceptive, temporal, cognitive.
2. Extract three entity sets:
   - O: objects mentioned in the question/options/answer
   - E: events/actions mentioned or implied by the question/options/answer
   - C: explicit cause-effect claims made by the answer or the selected option

Taxonomy:
{json.dumps(taxonomy, indent=2)}

Question:
{request.qa.render()}

LVM Answer:
{request.lvm_answer}

Return valid JSON only:
{{
  "hallucination_level": "perceptive|temporal|cognitive",
  "entities": {{
    "O": ["object 1"],
    "E": ["event 1"],
    "C": ["cause-effect claim 1"]
  }}
}}
""".strip()


def reasoning_prompt(request: DrVRequest, evidence: EvidenceBundle) -> str:
    evidence_json = json.dumps(_serialize_evidence(evidence), indent=2)
    return f"""
You are implementing Step 5 of Dr.V-Agent.
Judge whether the LVM answer contains hallucination by comparing it against the perceptive, temporal, and cognitive evidence.

Question:
{request.qa.render()}

LVM Answer:
{request.lvm_answer}

Evidence:
{evidence_json}

Return valid JSON only:
{{
  "has_hallucination": true,
  "error_points": ["short evidence-grounded diagnosis"],
  "confidence": 0.0
}}
""".strip()


def feedback_prompt(
    request: DrVRequest,
    evidence: EvidenceBundle,
    assessment: HallucinationAssessment,
) -> str:
    evidence_json = json.dumps(_serialize_evidence(evidence), indent=2)
    assessment_json = json.dumps(
        {
            "has_hallucination": assessment.has_hallucination,
            "error_points": assessment.error_points,
            "confidence": assessment.confidence,
        },
        indent=2,
    )
    return f"""
You are implementing Step 6 of Dr.V-Agent.
Generate structured feedback F = (A, R) for the target LVM.
- A must summarize the extracted spatial-temporal-causal evidence.
- R must give concise suggestions to revise the answer using that evidence.

Question:
{request.qa.render()}

LVM Answer:
{request.lvm_answer}

Evidence:
{evidence_json}

Assessment:
{assessment_json}

Return valid JSON only:
{{
  "feedback": {{
    "A": "summary of grounded evidence",
    "R": "revision guidance for the LVM"
  }}
}}
""".strip()


def _serialize_evidence(evidence: EvidenceBundle) -> dict:
    def serialize_bbox(box) -> dict:
        return {"x": box.x, "y": box.y, "w": box.w, "h": box.h}

    perceptive = {}
    for name, item in evidence.perceptive.items():
        perceptive[name] = {
            "consensus": [
                {
                    "timestamp": observation.timestamp,
                    "bbox": serialize_bbox(observation.bbox),
                    "confidence": observation.confidence,
                    "source": observation.source,
                }
                for observation in item.consensus
            ],
            "notes": item.notes,
        }

    temporal = {}
    for name, item in evidence.temporal.items():
        temporal[name] = {
            "consensus": [
                {
                    "start": interval.start,
                    "end": interval.end,
                    "confidence": interval.confidence,
                    "source": interval.source,
                }
                for interval in item.consensus
            ],
            "notes": item.notes,
        }

    cognitive = {}
    for name, item in evidence.cognitive.items():
        cognitive[name] = {
            "time_interval": (
                {
                    "start": item.time_interval.start,
                    "end": item.time_interval.end,
                    "confidence": item.time_interval.confidence,
                    "source": item.time_interval.source,
                }
                if item.time_interval
                else None
            ),
            "captions_by_source": item.captions_by_source,
            "consensus_caption": item.consensus_caption,
            "notes": item.notes,
        }
    return {"perceptive": perceptive, "temporal": temporal, "cognitive": cognitive}
