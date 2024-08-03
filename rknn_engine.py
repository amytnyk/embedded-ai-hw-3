from typing import Optional

import numpy as np

from rknnlite.api import RKNNLite

from engine_base import EngineBase
from engine_parameters import EngineParameters


class RknnEngine(EngineBase):
    def __init__(self, parameters: Optional[EngineParameters] = None):
        super().__init__(parameters, False)

        self._rknn_lite: RKNNLite = RKNNLite(verbose=False)
        self._rknn_lite.load_rknn(self._parameters.model_path)
        self._rknn_lite.init_runtime()

    def _process(self, image_data):
        outputs = self._rknn_lite.inference(inputs=[np.expand_dims(image_data, axis=0)])
        outputs[0] = np.squeeze(outputs[0])
        outputs[0] = np.expand_dims(outputs[0], axis=0)
        print(len(outputs), outputs[0].shape)
        return outputs

    def __del__(self):
        self._rknn_lite.release()
