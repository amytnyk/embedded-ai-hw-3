from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum, auto
from typing import Type

import cv2

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
    host_ip: str = "127.0.0.1"
    host_port: int = 5000


class YOLOv8Inference:
    def __init__(self, parameters: YOLOv8InferenceParameters):
        self._parameters: YOLOv8InferenceParameters = parameters

        self._video_capture: cv2.VideoCapture = cv2.VideoCapture(self._parameters.input_video_path)

        gst_out = (f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay "
                   f"config-interval=1 pt=96 ! "
                   f"udpsink host={self._parameters.host_ip} port={self._parameters.host_port} auto-multicast=true")
        self._output_writer: cv2.VideoWriter = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, 30,
                                                               self._get_next_frame().shape[:2][::-1], True)

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

    def _draw_overlay(self, frame, boxes):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = frame.copy()
        for box in boxes:
            cls = int(box[-1])
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3, 4)
            cv2.putText(image,
                        f"{self._parameters.detector_parameters.model_info.classes[cls]}:{round(box[4], 2)}",
                        (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        return image

    def _process_frame(self, frame):
        self._output_writer.write(self._draw_overlay(frame, self._detector.detect(frame)))

    def _update(self):
        self._process_frame(self._get_next_frame())


def main():
    parser = ArgumentParser(
        prog='YOLOv8 Inference',
        description='Run yolov8 model')
    parser.add_argument('--engine', type=str, choices=["rknn", "onnx"], required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--input-video-path', type=str, required=True)

    args = parser.parse_args()

    inference = YOLOv8Inference(YOLOv8InferenceParameters(
        detector_type=DetectorType.RKNN if args.engine == "rknn" else DetectorType.ONNX,
        detector_parameters=DetectorParameters(args.model_path),
        input_video_path=args.input_video_path
    ))
    inference.run()


if __name__ == "__main__":
    main()
