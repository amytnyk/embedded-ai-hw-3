import time
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum, auto
from typing import Type, List

import cv2
import numpy as np

from detection_result import DetectionResult
from detector_base import DetectorBase
from detector_parameters import DetectorParameters
from onnx_detector import OnnxDetector
from rknn_detector import RknnDetector


class DetectorType(Enum):
    ONNX = auto()
    RKNN = auto()


@dataclass
class YOLOv8InferenceParameters:
    detector_type: DetectorType
    detector_parameters: DetectorParameters
    input_video_path: str
    host_ip: str
    host_port: int


class YOLOv8Inference:
    def __init__(self, parameters: YOLOv8InferenceParameters):
        self._parameters: YOLOv8InferenceParameters = parameters

        self._video_capture: cv2.VideoCapture = cv2.VideoCapture(self._parameters.input_video_path)
        if self._video_capture.isOpened() is False:
            raise RuntimeError("Cannot open video capture")
        self._output_writer: cv2.VideoWriter = cv2.VideoWriter(
            f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self._parameters.host_ip} port={self._parameters.host_port}",
            cv2.CAP_GSTREAMER, 0, 30,
            tuple(self._get_next_frame().shape[:2][::-1]), True)
        self._color_palette = np.random.uniform(0, 255,
                                                size=(len(self._parameters.detector_parameters.model_info.classes), 3))

        self._detector: DetectorBase = self._get_detector_by_type(self._parameters.detector_type)(
            self._parameters.detector_parameters)

    def run(self):
        while True:
            self._update()

    @staticmethod
    def _get_detector_by_type(detector_type: DetectorType) -> Type[DetectorBase]:
        if detector_type == DetectorType.ONNX:
            return OnnxDetector
        elif detector_type == DetectorType.RKNN:
            return RknnDetector

    def _get_next_frame(self):
        ret, frame = self._video_capture.read()
        if ret:
            return frame
        raise RuntimeError("Cannot read frame")

    def _draw_overlay(self, frame, detections: List[DetectionResult]):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = frame.copy()
        for detection in detections:
            x1, y1, w, h = detection.box

            color = self._color_palette[detection.class_id]

            cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            label = (f"{self._parameters.detector_parameters.model_info.classes[detection.class_id]}: "
                     f"{detection.score:.2f}")

            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            cv2.rectangle(
                image, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                cv2.FILLED
            )

            cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return image

    def _process_frame(self, frame):
        start_time = time.time()
        detections = self._detector.detect(frame)
        print("Inference time: {:.1f}ms".format((time.time() - start_time) * 1000))
        self._output_writer.write(self._draw_overlay(frame, detections))

    def _update(self):
        self._process_frame(self._get_next_frame())


def main():
    parser = ArgumentParser(
        prog='YOLOv8 Inference',
        description='Run yolov8 model')
    parser.add_argument('--engine', type=str, choices=["rknn", "onnx"], required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--input-video-path', type=str, required=True)
    parser.add_argument('--host-ip', type=str, required=True)
    parser.add_argument('--host-port', type=int, required=True)

    args = parser.parse_args()

    inference = YOLOv8Inference(YOLOv8InferenceParameters(
        detector_type=DetectorType.RKNN if args.engine == "rknn" else DetectorType.ONNX,
        detector_parameters=DetectorParameters(args.model_path),
        input_video_path=args.input_video_path,
        host_ip=args.host_ip,
        host_port=args.host_port
    ))
    inference.run()


if __name__ == "__main__":
    main()
