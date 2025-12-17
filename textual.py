import json
import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


class TextToolkit:
    def __init__(self):
        self.gpt4o_client = OpenAI(api_key=OPENAI_API_KEY)
        self.deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
        )

    def _clean_and_parse_json(self, content: str) -> Dict:
        try:
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def gpt4o_hallucination_classify(self, qa_pair: Dict, lvm_answer: str) -> Dict:
        prompt = f"""
        Task: Analyze the video QA pair and LVM answer to complete two goals:
        1. Classify hallucination level (only choose one: perceptive/temporal/cognitive):
           - Perceptive: Errors in object/color/number recognition, OCR, or static spatial relations.
           - Temporal: Errors in event sequence, dynamic attributes (speed/direction), or action order.
           - Cognitive: Errors in causal reasoning, counterfactual prediction, or common sense.
        2. Extract entities (list all relevant items):
           - O (Objects): Physical items mentioned (e.g., "red car", "baby").
           - E (Events): Dynamic actions mentioned (e.g., "eat snack", "walk to shelf").
           - C (Causal Claims): Cause-effect relationships (e.g., "baby cried because mom blamed him").
        
        QA Pair: {qa_pair}
        LVM Answer: {lvm_answer}
        
        Output ONLY valid JSON (no extra text):
        {{
            "hallucination_level": "str",
            "entities": {{
                "O": ["str1", "str2"],
                "E": ["str1", "str2"],
                "C": ["str1", "str2"]
            }}
        }}
        """
        response = self.gpt4o_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return self._clean_and_parse_json(content)

    def deepseek_reasoning_verify(self, evidence: Dict, lvm_answer: str) -> Dict:
        prompt = f"""
        Task: Judge if the LVM answer has hallucinations by comparing with video evidence.
        
        Verification Evidence: {evidence}
        LVM Answer: {lvm_answer}
        
        Output ONLY valid JSON:
        {{
            "has_hallucination": true/false,
            "error_points": ["Error description", ...],
            "confidence": 0.0-1.0
        }}
        """
        response = self.deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return self._clean_and_parse_json(content)

    def gpt4o_feedback_generate(
        self, evidence: Dict, hallucination_assessment: Dict
    ) -> Dict:
        prompt = f"""
        Task: Generate structured feedback for LVM to fix hallucinations.
        
        Verification Evidence: {evidence}
        Hallucination Assessment: {hallucination_assessment}
        
        Output ONLY valid JSON:
        {{
            "feedback": {{
                "A": "Summary of spatial-temporal-causal evidence",
                "R": "Correction recommendations"
            }}
        }}
        """
        response = self.gpt4o_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return self._clean_and_parse_json(content)
