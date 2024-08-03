# Note: partially taken from https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/yolov8.py

from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
from torch import tensor

from detection_result import DetectionResult
from engine_parameters import EngineParameters


class EngineBase(ABC):
    def __init__(self, parameters: EngineParameters, do_preprocess_normalization: bool = True):
        self._parameters: EngineParameters = parameters
        self._do_preprocess_normalization: bool = do_preprocess_normalization

    @abstractmethod
    def _process(self, image_data):
        pass

    def detect(self, image) -> List[DetectionResult]:
        image_data = self._preprocess(image)
        output = self._process(image_data)
        return self._postprocess(output, image)

    def _preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(img,
                                (self._parameters.model_info.image_width, self._parameters.model_info.image_height))

        if self._do_preprocess_normalization:
            image_data = np.array(image_data) / 255.0
            image_data = np.transpose(image_data, (2, 0, 1))
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def _postprocess(self, output, image):
        boxes, scores, classes_conf = [], [], []
        default_branch = 3
        pair_per_branch = len(output) // default_branch

        for i in range(default_branch):
            boxes.append(self._box_process(output[pair_per_branch * i]))
            classes_conf.append(output[pair_per_branch * i + 1])
            scores.append(np.ones_like(output[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self._nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return []

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        def normalize_box(box):
            x_factor = image.shape[1] / self._parameters.model_info.image_width
            y_factor = image.shape[0] / self._parameters.model_info.image_height
            sx, sy, ex, ey = map(int, box)
            return [int(sx * x_factor), int(sy * y_factor), int((ex - sx) * x_factor), int((ey - sy) * y_factor)]

        return [DetectionResult(normalize_box(boxes[i]), float(scores[i]), int(classes[i])) for i in
                range(min(len(boxes), len(classes), len(scores)))]

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self._parameters.conf_threshold)
        scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._parameters.iou_threshold)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    @staticmethod
    def _dfl(position):
        x = tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y.numpy()

    def _box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self._parameters.model_info.image_height // grid_h,
                           self._parameters.model_info.image_width // grid_w]).reshape(1, 2, 1, 1)

        position = self._dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy
