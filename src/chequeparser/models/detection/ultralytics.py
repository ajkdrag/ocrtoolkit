from typing import Any, List

import numpy as np

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

    def __init__(self, arch, path, arch_config=None):
        self.path = path
        self.arch = arch
        self.arch_config = arch_config if arch_config else {}

        self.loader = arch(**self.arch_config)
        self.network = self.loader(path)

    def predict_batch(
        self, images: List[np.ndarray], **kwargs
    ) -> List[DetectionResults]:
        l_preds = self.network.predict(images, **kwargs)
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
