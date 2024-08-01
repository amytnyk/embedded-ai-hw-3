from detector_base import DetectorBase
from detector_parameters import DetectorParameters


class OnnxDetector(DetectorBase):
    def __init__(self, parameters: DetectorParameters):
        super().__init__(parameters)

    def detect(self, image):
        pass
