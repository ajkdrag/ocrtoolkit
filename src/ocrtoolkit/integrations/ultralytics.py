from typing import List, Optional

import numpy as np

from ocrtoolkit.wrappers.bbox import BBox
from ocrtoolkit.wrappers.detection_results import DetectionResults
from ocrtoolkit.wrappers.model import DetectionModel


class UltralyticsModel(DetectionModel):
    valid_kwargs = set(["verbose", "stream", "device"])

    def _predict(self, images: List[np.ndarray], **kwargs) -> List[DetectionResults]:
        l_preds = self.model.predict(images, **kwargs)
        l_results = []
        for image, preds in zip(images, l_preds):
            np_preds = preds.cpu().numpy()
            d_names = np_preds.names
            l_coords = np_preds.boxes.xyxy.tolist()
            l_confs = np_preds.boxes.conf.tolist()
            l_labels = [d_names[int(i)] for i in np_preds.boxes.cls]
            l_bboxes = [
                BBox(*bbox, conf=conf, label=label, normalized=False)
                for bbox, conf, label in zip(l_coords, l_confs, l_labels)
            ]
            l_results.append(
                DetectionResults(l_bboxes, width=image.shape[1], height=image.shape[0])
            )
        return l_results


def load(
    model_name: str, path: Optional[str], device: str, model_kwargs: dict, **kwargs
):
    """Factory method to load model."""
    if model_name == "yolov8":
        from ultralytics import YOLO

        path = "yolov8n.pt" if path is None else path
        model = YOLO(path, **model_kwargs)
        return UltralyticsModel(model, path, device)
    elif model_name == "rtdetr":
        from ultralytics import RTDETR

        path = "rtdetr-l.pt" if path is None else path
        model = RTDETR(path, **model_kwargs)
        return UltralyticsModel(model, path, device)
    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")
