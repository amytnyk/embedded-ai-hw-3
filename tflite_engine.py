import numpy as np

from engine_parameters import EngineParameters
import tflite_runtime.interpreter as tflite

from engine_base import EngineBase


class TFLiteEngine(EngineBase):
    def __init__(self, parameters: EngineParameters):
        super().__init__(parameters, True)

        self._interpreter = tflite.Interpreter(model_path=self._parameters.model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        print(self._input_details, self._output_details)

    def _process(self, image_data):
        print(image_data.shape)
        self._interpreter.set_tensor(self._input_details[0]['index'], np.transpose(image_data, (0, 2, 3, 1)))
        self._interpreter.invoke()

        output = [
            np.transpose(np.expand_dims(self._interpreter.get_tensor(self._output_details[i]['index'])[0], axis=0),
                         (0, 3, 1, 2)) for i in
            range(len(self._output_details))]
        print(len(output), [output[i].shape for i in range(len(output))])
        return output
