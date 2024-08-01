from abc import ABC, abstractmethod

from detector_parameters import DetectorParameters


class DetectorBase(ABC):
    def __init__(self, parameters: DetectorParameters):
        self._parameters: DetectorParameters = parameters

    @abstractmethod
    def detect(self, image):
        pass
