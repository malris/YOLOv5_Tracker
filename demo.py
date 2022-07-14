from typing import Union, Optional, List

from tracker import Tracker, get_centroid_detections
from utils import Video, draw_trajectories, draw_bboxes
import torch
import yolov5
import argparse
import cv2

import numpy as np


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = yolov5.load(model_path, device=device)

    def __call__(
            self, img: Union[str, np.ndarray], conf_threshold: float = 0.25,
            iou_threshold: float = 0.45, image_size: int = 720,
    ) -> torch.tensor:
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

        detections = self.model(img, size=image_size)
        return detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("--source", type=str, nargs="+", help="Paths to the files to be processed")
parser.add_argument("--weights", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img-size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf-thres", type=float, default="0.5", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou-thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--output-path", type=str, default=None, help="Output video path")
args = parser.parse_args()

model = YOLO(args.weights, device=args.device)

for input_path in args.source:
    video = Video(input_path=input_path, output_path=args.output_path)

    tracker = Tracker(distance_threshold=30, disappearing_threshold=25)

    for frame in video:
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thresh,
            image_size=args.img_size
        )

        detections = get_centroid_detections(yolo_detections)
        tracked_objects = tracker.update(detections=detections)

        draw_bboxes(frame, yolo_detections, model.model.names)

        draw_trajectories(frame, tracked_objects)

        cv2.imshow('Result', frame)
        cv2.waitKey(1)

        if args.output_path is not None:
            video.write(frame)
