from engine_parameters import EngineParameters
import onnxruntime

from engine_base import EngineBase


class OnnxEngine(EngineBase):
    def __init__(self, parameters: EngineParameters):
        super().__init__(parameters, True)

        self._sess = onnxruntime.InferenceSession(self._parameters.model_path, providers=["CPUExecutionProvider"])
        self._input_list = [self._sess.get_inputs()[i].name for i in range(len(self._sess.get_inputs()))]
        self._output_list = [self._sess.get_outputs()[i].name for i in range(len(self._sess.get_outputs()))]

    def _process(self, image_data):
        return self._sess.run(self._output_list, {self._sess.get_inputs()[0].name: image_data})
