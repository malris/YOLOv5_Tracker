from typing import Union, Optional

import yolov5
import numpy as np
import torch


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
