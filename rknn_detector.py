from typing import Optional

import numpy as np

from detector_base import DetectorBase
from rknnlite.api import RKNNLite

from detector_parameters import DetectorParameters


class RknnDetector(DetectorBase):
    def __init__(self, parameters: Optional[DetectorParameters] = None):
        super().__init__(parameters)
        self._rknn_lite: RKNNLite = RKNNLite(verbose=False)
        self._rknn_lite.load_rknn(self._parameters.model_path)
        self._rknn_lite.init_runtime()

    def _process(self, image_data):
        outputs = self._rknn_lite.inference(inputs=[image_data])
        outputs[0] = np.squeeze(outputs[0])
        outputs[0] = np.expand_dims(outputs[0], axis=0)
        return outputs

    def __del__(self):
        self._rknn_lite.release()
