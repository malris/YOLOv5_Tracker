from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import torch


class Detection:
    def __init__(self, points: np.array, scores=None):
        self.points = points
        self.scores = scores


class TrackedObject:
    def __init__(self, initial_detection: Detection, past_detections_length: int = 0):
        self.last_detection = initial_detection
        self.past_detections_length = past_detections_length
        self.past_detections = [initial_detection] if past_detections_length > 0 else []

    def hit(self, detection: Detection):
        self.add_to_past_detections(detection)
        self.last_detection = detection

    def add_to_past_detections(self, detection: Detection):
        if self.past_detections_length == 0:
            return

        if len(self.past_detections) >= self.past_detections_length:
            self.past_detections.pop(0)

        self.past_detections.append(detection)


class Tracker:
    def __init__(self, distance_threshold: float, disappearing_threshold: int = 50, past_detections_length: int = 50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.disappearing_threshold = disappearing_threshold
        self.distance_threshold = distance_threshold

        self.past_detections_length = past_detections_length

    def register(self, detection: Detection):
        self.objects[self.next_object_id] = TrackedObject(detection, self.past_detections_length)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections: list[Detection]):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] > self.disappearing_threshold:
                    self.deregister(obj_id)

            return self.objects

        input_centroids = np.array([det.points for det in detections], dtype="int")

        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([obj.last_detection.points for obj in self.objects.values()], dtype="int")

            dist_matrix = dist.cdist(np.array(object_centroids), input_centroids)

            rows = dist_matrix.min(axis=1).argsort()
            cols = dist_matrix.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if dist_matrix[row][col] >= self.distance_threshold:
                    used_rows.add(row)
                    used_cols.add(col)
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id].hit(detections[col])
                self.disappeared[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dist_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, dist_matrix.shape[1])).difference(used_cols)

            if dist_matrix.shape[0] >= dist_matrix.shape[1]:
                for row in unused_rows:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1

                    if self.disappeared[obj_id] > self.disappearing_threshold:
                        self.deregister(obj_id)

            else:
                for col in unused_cols:
                    self.register(detections[col])

        return self.objects


def get_centroid_detections(yolo_detections: torch.tensor) -> list[Detection]:
    centroid_detections = []

    detections_as_xywh = yolo_detections.xywh[0]
    for detection_as_xywh in detections_as_xywh:
        centroid = np.array(
            [
                detection_as_xywh[0].item(),
                detection_as_xywh[1].item()
            ]
        )
        scores = np.array([detection_as_xywh[4].item()])
        centroid_detections.append(
            Detection(points=centroid, scores=scores)
        )

    return centroid_detections
