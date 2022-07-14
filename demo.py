from tracker.tracker import Tracker, get_centroid_detections
from utils.video import Video
from utils.drawing import draw_trajectories, draw_bboxes
from model.yolo_model import YOLO

import argparse
import cv2

parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("--source", type=str, nargs="+", help="Paths to the files to be processed")
parser.add_argument("--weights", type=str, default="yolov5m6.pt", help="YOLOv5 models path")
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
