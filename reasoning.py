import json
from typing import Dict

from cognitive import CognitiveToolkit
from perceptive import PerceptiveToolkit
from preprocessing import VideoPreprocessor
from temporal import TemporalToolkit
from textual import TextToolkit


class DrVAgent:
    def __init__(self, tool_config: Dict):
        self.text_toolkit = TextToolkit()
        self.perceptive_toolkit = PerceptiveToolkit(
            sam2_weight_path=tool_config["sam2_weight"],
            yolo_weight_path=tool_config["yolo_weight"],
        )
        self.temporal_toolkit = TemporalToolkit(
            cg_stvg_weight_path=tool_config["cg_stvg_weight"],
            grounded_videollm_weight_path=tool_config["grounded_videollm_weight"],
        )
        self.cognitive_toolkit = CognitiveToolkit(
            internvl2_weight_path=tool_config["internvl2_weight"],
            qwen2vl_weight_path=tool_config["qwen2vl_weight"],
        )
        self.video_preprocessor = None

    def run(self, video_path: str, qa_pair: Dict, lvm_answer: str) -> Dict:
        self.video_preprocessor = VideoPreprocessor(video_path=video_path)
        frames, frame_timestamps = self.video_preprocessor.process()

        evidence = {
            "perceptive": {},
            "temporal": {},
            "cognitive": {},
        }

        classify_result = self.text_toolkit.gpt4o_hallucination_classify(
            qa_pair, lvm_answer
        )
        hallucination_level = classify_result.get("hallucination_level", "perceptive")
        entities = classify_result.get("entities", {"O": [], "E": [], "C": []})

        if entities.get("O"):
            perceptive_evidence = {}
            for obj in entities["O"]:
                det_result = self.perceptive_toolkit.cross_validate_object_detection(
                    frames=frames, frame_timestamps=frame_timestamps, object_name=obj
                )
                perceptive_evidence[obj] = det_result
            evidence["perceptive"] = perceptive_evidence

        if hallucination_level in ["temporal", "cognitive"] and entities.get("E"):
            temporal_evidence = {}
            for event in entities["E"]:
                ground_result = self.temporal_toolkit.cross_validate_event_grounding(
                    video_path=video_path, event_name=event
                )
                temporal_evidence[event] = ground_result
            evidence["temporal"] = temporal_evidence

        if hallucination_level == "cognitive" and entities.get("C"):
            cognitive_evidence = {}
            for causal_claim in entities["C"]:
                target_event_key = None
                for e_key in evidence.get("temporal", {}).keys():
                    if e_key in causal_claim:
                        target_event_key = e_key
                        break

                time_interval = None
                if target_event_key:
                    intervals = evidence["temporal"][target_event_key].get(
                        "time_intervals", []
                    )
                    if intervals:
                        time_interval = intervals[0]

                if not time_interval:
                    cognitive_evidence[causal_claim] = {
                        "error": "No temporal grounding found for claim"
                    }
                    continue

                causal_caption = self.cognitive_toolkit.generate_causal_caption(
                    frames=frames,
                    time_interval=time_interval,
                    frame_timestamps=frame_timestamps,
                )

                common_sense_result = self.cognitive_toolkit.verify_common_sense(
                    causal_caption=causal_caption,
                    video_context=f"Event '{target_event_key}' at interval {time_interval}",
                )

                cognitive_evidence[causal_claim] = {
                    "generated_caption": causal_caption,
                    "common_sense_check": common_sense_result,
                }
            evidence["cognitive"] = cognitive_evidence

        reasoning_result = self.text_toolkit.deepseek_reasoning_verify(
            evidence, lvm_answer
        )

        feedback_result = self.text_toolkit.gpt4o_feedback_generate(
            evidence, reasoning_result
        )

        return {
            "hallucination_classify": classify_result,
            "verification_evidence": evidence,
            "hallucination_assessment": reasoning_result,
            "structured_feedback": feedback_result,
        }


def main():
    tool_config = {
        "sam2_weight": "xxxxxx",
        "yolo_weight": "xxxxxx",
        "cg_stvg_weight": "xxxxxx",
        "grounded_videollm_weight": "xxxxxx",
    }

    test_video_path = "./test_video.mp4"
    test_qa_pair = {
        "question": "Why did the baby walk to the shelf?",
        "options": ["A. Play", "B. Toy", "C. Book", "D. Rainbow"],
    }
    test_lvm_answer = "The baby walked to the shelf to play with dad."

    drv_agent = DrVAgent(tool_config=tool_config)
    print("Agent initialized.")


if __name__ == "__main__":
    main()
