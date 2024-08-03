from dataclasses import dataclass

from yolo_model_info import YOLOModelInfo


@dataclass
class EngineParameters:
    model_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    model_info: YOLOModelInfo = YOLOModelInfo()
