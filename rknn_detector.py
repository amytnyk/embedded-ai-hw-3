from dataclasses import dataclass
from typing import Optional

import numpy as np

from detector_base import DetectorBase
from rknnlite.api import RKNNLite

from detector_parameters import DetectorParameters
from utils import preprocess, postprocess


class RknnDetector(DetectorBase):
    def __init__(self, parameters: Optional[DetectorParameters] = None):
        super().__init__(parameters)
        self._rknn_lite: RKNNLite = RKNNLite(verbose=False)
        self._rknn_lite.load_rknn(self._parameters.model_path)
        self._rknn_lite.init_runtime()

    def detect(self, image):
        image_4c, image_3c = preprocess(image, self._parameters.model_info.image_height,
                                        self._parameters.model_info.image_width)
        self._rknn_lite.init_runtime()
        outputs = self._rknn_lite.inference(inputs=[image_3c])
        outputs[0] = np.squeeze(outputs[0])
        outputs[0] = np.expand_dims(outputs[0], axis=0)
        results = postprocess(outputs, image_4c, image_3c, self._parameters.conf_threshold,
                              self._parameters.iou_threshold)[0]
        return results[0] if isinstance(results[0], np.ndarray) else None

    def __del__(self):
        self._rknn_lite.release()
