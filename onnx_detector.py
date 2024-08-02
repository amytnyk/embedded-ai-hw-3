from detector_base import DetectorBase
from detector_parameters import DetectorParameters
import onnxruntime


class OnnxDetector(DetectorBase):
    def __init__(self, parameters: DetectorParameters):
        super().__init__(parameters, True)
        self._sess = onnxruntime.InferenceSession(self._parameters.model_path, providers=["CPUExecutionProvider"])
        self._input_list = [self._sess.get_inputs()[i].name for i in range(len(self._sess.get_inputs()))]
        self._output_list = [self._sess.get_outputs()[i].name for i in range(len(self._sess.get_outputs()))]

    def _process(self, image_data):
        return self._sess.run(self._output_list, {self._sess.get_inputs()[0].name: image_data})
