import numpy as np
from typing import Any, List
from loguru import logger
from chequeparser.models.arch import BaseArch
from chequeparser.models.detection.base import BaseDetect
from chequeparser.wrappers.bbox import BBox
from chequeparser.wrappers.detection_results import DetectionResults


class UltralyticsDetect(BaseDetect):
    path: str
    arch: BaseArch
    loader: Any
    network: Any
    arch_config: dict
    predict_config: dict

    def __init__(self, path, arch, arch_config=None, predict_config=None):
        self.path = path
        self.arch = arch
        self.arch_config = arch_config if arch_config else {}
        self.predict_config = predict_config if predict_config else {}

        self.loader = arch(**self.arch_config)
        self.network = self.loader(path)

    def predict(self, 
                image: np.ndarray) -> DetectionResults:
        return self.predict_batch([image])[0]
        
    def predict_batch(self, 
                      images: List[np.ndarray]) -> List[DetectionResults]:
        l_preds = self.network.predict(images, **self.predict_config)
        l_results = []
        for image, preds in zip(images, l_preds):
            np_preds = preds.cpu().numpy()
            d_names = np_preds.names
            l_coords = np_preds.boxes.xyxy.tolist()
            l_confs = np_preds.boxes.conf.tolist()
            l_labels = [d_names[int(i)] for i in np_preds.boxes.cls]
            l_bboxes = [
                BBox(*bbox, conf=conf, label=label, normalized=True)
                for bbox, conf, label in zip(l_coords, l_confs, l_labels)
            ]
            l_results.append(DetectionResults(l_bboxes, image))
        return l_results

