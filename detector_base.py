from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np

from detection_result import DetectionResult
from detector_parameters import DetectorParameters


class DetectorBase(ABC):
    def __init__(self, parameters: DetectorParameters):
        self._parameters: DetectorParameters = parameters

    @abstractmethod
    def _process(self, image_data):
        pass

    def detect(self, image) -> List[DetectionResult]:
        image_data = self._preprocess(image)
        output = self._process(image_data)
        return self._postprocess(output, image)

    def _preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._parameters.model_info.image_width, self._parameters.model_info.image_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def _postprocess(self, output, image):
        outputs = np.squeeze(output[0])
        print(outputs.shape)
        outputs = np.expand_dims(outputs, axis=0)
        outputs = np.transpose(outputs)

        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        x_factor = image.shape[1] / self._parameters.model_info.image_width
        y_factor = image.shape[0] / self._parameters.model_info.image_height

        for i in range(rows):
            classes_scores = outputs[i][4:]

            max_score = np.amax(classes_scores)

            if max_score >= self._parameters.conf_threshold:
                class_id = np.argmax(classes_scores)

                # print(outputs[i])

                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self._parameters.conf_threshold, self._parameters.iou_threshold)

        return [DetectionResult(boxes[i], scores[i], class_ids[i]) for i in indices]
