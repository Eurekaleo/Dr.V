from typing import Dict

from cg_stvg.model import CG_STVG
from grounded_videollm.inference import GroundedVideoLLMInferencer


class TemporalToolkit:
    def __init__(self, cg_stvg_weight_path: str, grounded_videollm_weight_path: str):
        self.cg_stvg = CG_STVG(weight_path=cg_stvg_weight_path).cuda()
        self.grounded_videollm = GroundedVideoLLMInferencer(
            model_path=grounded_videollm_weight_path, device="cuda"
        )

    def cross_validate_event_grounding(self, video_path: str, event_name: str) -> Dict:
        cg_stvg_result = self.cg_stvg.infer(
            video_path=video_path, text_query=event_name, threshold=0.3
        )
        grounded_vllm_result = self.grounded_videollm.temporal_grounding(
            video_path=video_path, event_description=event_name
        )

        final_intervals = []
        if (
            cg_stvg_result.get("intervals")
            and grounded_vllm_result.get("intervals")
            and len(cg_stvg_result["intervals"]) > 0
            and len(grounded_vllm_result["intervals"]) > 0
        ):
            cg_interval = cg_stvg_result["intervals"][0]
            vllm_interval = grounded_vllm_result["intervals"][0]

            t_start = max(cg_interval[0], vllm_interval[0])
            t_end = min(cg_interval[1], vllm_interval[1])

            if t_end > t_start:
                final_intervals.append([t_start, t_end])

        conf1 = cg_stvg_result.get("conf", 0.5)
        conf2 = grounded_vllm_result.get("conf", 0.5)

        return {
            "time_intervals": final_intervals,
            "confidence": (conf1 + conf2) / 2,
        }
