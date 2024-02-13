from typing import Any, List

import numpy as np
from loguru import logger

from doctr.models.predictor.base import _OCRPredictor

from chequeparser.models.arch import BaseArch
from chequeparser.models.detection.base import BaseDetect
from chequeparser.wrappers.bbox import BBox
from chequeparser.wrappers.detection_results import DetectionResults


class DoctrDetect(BaseDetect):
    path: str
    arch: BaseArch
    loader: Any
    network: Any
    arch_config: dict
    predict_config: dict

    def __init__(self, arch, path=None, arch_config=None, predict_config=None):
        self.path = path
        self.arch = arch
        self.arch_config = arch_config if arch_config else {}
        self.predict_config = predict_config if predict_config else {}

        self.loader = arch(**self.arch_config)
        self.network = self.loader(path)
        self.doctr_base_predictor = _OCRPredictor()

    def predict(self, image: np.ndarray) -> DetectionResults:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: List[np.ndarray]) -> List[DetectionResults]:
        l_preds = self.network(images, **self.predict_config)
        l_loc_preds = [list(loc_pred.values())[0] 
                       for loc_pred in l_preds]
        l_loc_preds = (self.doctr_base_predictor
                           ._remove_padding(images, l_loc_preds))

        l_results = []
        for image, preds in zip(images, l_loc_preds):
            l_confs = preds[:, 4].tolist()
            l_labels = [0]*len(l_confs)
            l_coords = preds[:, :4].tolist()
            l_bboxes = [
                BBox(*bbox, conf=conf, label=label, normalized=True)
                for bbox, conf, label in zip(l_coords, l_confs, l_labels)
            ]
            l_results.append(DetectionResults(l_bboxes, image))
        return l_results
