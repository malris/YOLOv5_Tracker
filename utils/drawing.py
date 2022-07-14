import numpy as np

import torch
import cv2
from yolov5.utils.plots import Annotator


class Color:
    green = (0, 128, 0)
    white = (255, 255, 255)
    olive = (0, 128, 128)
    black = (0, 0, 0)
    navy = (128, 0, 0)
    red = (0, 0, 255)


def draw_bboxes(frame: np.array, predictions: torch.Tensor, names):
    frame_scale = frame.shape[0] / 100
    thickness = int(max(frame_scale / 7, 1))
    color = Color.red

    predictions = predictions.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    for box, score, cat in zip(boxes, scores, categories):
        annotator = Annotator(frame, line_width=thickness, example=str(names))
        c = int(cat)
        label = f'{names[c]} {score:.2f}'
        annotator.box_label(box, label, color=color)

        frame = annotator.result()


def draw_trajectories(frame: np.array, tracked_objects: dict, period: int = 5):
    frame_scale = frame.shape[0] / 100
    radius = int(max(frame_scale * 0.5, 1))
    thickness = int(max(frame_scale / 5, 1))

    color = Color.red
    for obj_id, obj in tracked_objects.items():
        points = np.array([det.points for det in obj.past_detections], dtype='int')
        points = np.unique(points, axis=0)

        row_n = points.shape[0]
        markers = points[row_n - 1::-period, :]

        for marker in markers:
            cv2.circle(
                frame,
                tuple(marker.astype(int)),
                radius=radius,
                color=color,
                thickness=thickness,
            )

        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, color, thickness=thickness)
