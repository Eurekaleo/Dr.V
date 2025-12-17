import json
from typing import Dict, List

import numpy as np
from internvl import InternVLModel
from qwen_vl import QwenVL


class CognitiveToolkit:
    def __init__(self, internvl2_weight_path: str, qwen2vl_weight_path: str):
        self.internvl2 = InternVLModel(
            model_path=internvl2_weight_path, device="cuda", vit_model="vit-g-14"
        )
        self.qwen2vl = QwenVL(model_path=qwen2vl_weight_path, device="cuda")

    def generate_causal_caption(
        self,
        frames: List[np.ndarray],
        time_interval: List[float],
        frame_timestamps: List[float],
    ) -> str:
        interval_frames = []
        for frame, ts in zip(frames, frame_timestamps):
            if time_interval[0] <= ts <= time_interval[1]:
                interval_frames.append(frame)

        if len(interval_frames) == 0:
            return "No frames detected in the target time interval"

        if len(interval_frames) > 8:
            indices = np.linspace(0, len(interval_frames) - 1, 8, dtype=int)
            interval_frames = [interval_frames[i] for i in indices]

        caption = self.internvl2.generate_caption(
            images=interval_frames,
            prompt="Describe the causal relationship in the video (format: 'cause → effect')",
        )
        return caption

    def verify_common_sense(self, causal_caption: str, video_context: str) -> Dict:
        prompt = f"""
        Task: Judge if the causal relationship is consistent with common sense and explain.
        Causal Relationship: {causal_caption}
        Video Context: {video_context}
        
        Output ONLY valid JSON:
        {{
            "is_consistent": true/false,
            "reason": "Explanation"
        }}
        """
        result = self.qwen2vl.generate(text=prompt, temperature=0.2)
        try:
            clean_result = result.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_result)
        except Exception:
            return {"is_consistent": False, "reason": "Output parsing failed"}
