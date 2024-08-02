from dataclasses import dataclass
from typing import List


@dataclass
class DetectionResult:
    box: List[int]
    score: float
    class_id: int
