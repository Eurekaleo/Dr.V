from typing import Dict, List

import numpy as np
from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO


class PerceptiveToolkit:
    def __init__(self, sam2_weight_path: str, yolo_weight_path: str):
        self.grounded_dino_model = load_model(
            model_config_path="xxxxxx",
            model_checkpoint_path="xxxxxx",
        )
        self.sam2 = sam_model_registry["hiera_large"](
            checkpoint=sam2_weight_path
        ).cuda()
        self.sam2_predictor = SamPredictor(self.sam2)
        self.yolo_world = YOLO(yolo_weight_path).cuda()

    def cross_validate_object_detection(
        self, frames: List[np.ndarray], frame_timestamps: List[float], object_name: str
    ) -> Dict:
        yolo_results = []
        try:
            self.yolo_world.set_classes([object_name])
        except Exception:
            pass

        for frame in frames:
            result = self.yolo_world.predict(
                frame,
                conf=0.1,
                verbose=False,
                imgsz=640,
            )[0]

            if result.boxes is not None and len(result.boxes) > 0:
                box = result.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                yolo_results.append(
                    {
                        "bbox": (x1, y1, w, h),
                        "conf": result.boxes.conf[0].item(),
                    }
                )
            else:
                yolo_results.append(None)

        sam2_results = []
        for frame in frames:
            boxes, logits, _ = predict(
                model=self.grounded_dino_model,
                image=frame,
                caption=object_name,
                box_threshold=0.3,
                text_threshold=0.25,
            )

            if len(boxes) > 0:
                H, W, _ = frame.shape
                box_norm = boxes[0].cpu().numpy()
                cx, cy, w_norm, h_norm = box_norm

                cx_px = cx * W
                cy_px = cy * H
                w_px = w_norm * W
                h_px = h_norm * H

                x1_box = cx_px - (w_px / 2)
                y1_box = cy_px - (h_px / 2)
                x2_box = cx_px + (w_px / 2)
                y2_box = cy_px + (h_px / 2)

                input_box = np.array([x1_box, y1_box, x2_box, y2_box])

                self.sam2_predictor.set_image(frame)
                masks, _, _ = self.sam2_predictor.predict(
                    box=input_box, multimask_output=False
                )
                mask = masks[0].astype(bool)

                if mask.any():
                    y, x = np.where(mask)
                    x1, x2 = x.min(), x.max()
                    y1, y2 = y.min(), y.max()
                    sam2_results.append(
                        {
                            "bbox": (
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                            ),
                            "conf": float(logits[0]),
                        }
                    )
                else:
                    sam2_results.append(None)
            else:
                sam2_results.append(None)

        final_results = {"bboxes": [], "timestamps": [], "confidence": []}

        for i, (yolo_res, sam2_res) in enumerate(zip(yolo_results, sam2_results)):
            if yolo_res is not None and sam2_res is not None:
                x1_y, y1_y, w_y, h_y = yolo_res["bbox"]
                x1_s, y1_s, w_s, h_s = sam2_res["bbox"]

                x1 = max(x1_y, x1_s)
                y1 = max(y1_y, y1_s)
                x2 = min(x1_y + w_y, x1_s + w_s)
                y2 = min(y1_y + h_y, y1_s + h_s)

                if x2 > x1 and y2 > y1:
                    final_results["bboxes"].append((x1, y1, x2 - x1, y2 - y1))
                    final_results["timestamps"].append(frame_timestamps[i])
                    final_results["confidence"].append(
                        (yolo_res["conf"] + sam2_res["conf"]) / 2
                    )

        return final_results
