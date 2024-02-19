from typing import Any, List

import numpy as np
from doctr.models.predictor.base import _OCRPredictor
from loguru import logger

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

    def __init__(self, arch, path=None, arch_config=None):
        self.path = path
        self.arch = arch
        self.arch_config = arch_config if arch_config else {}

        self.loader = arch(**self.arch_config)
        self.network = self.loader(path)
        self.doctr_base_predictor = _OCRPredictor()

    def predict_batch(
        self, images: List[np.ndarray], **kwargs
    ) -> List[DetectionResults]:
        l_preds = self.network(images, **kwargs)
        l_loc_preds = [list(loc_pred.values())[0] for loc_pred in l_preds]
        l_loc_preds = self.doctr_base_predictor._remove_padding(images, l_loc_preds)

        l_results = []
        for image, preds in zip(images, l_loc_preds):
            l_confs = preds[:, 4].tolist()
            l_labels = ["0"] * len(l_confs)
            l_coords = preds[:, :4].tolist()
            l_bboxes = [
                BBox(*bbox, conf=conf, label=label, normalized=True).denormalize(
                    image.shape[1], image.shape[0]
                )
                for bbox, conf, label in zip(l_coords, l_confs, l_labels)
            ]
            l_results.append(
                DetectionResults(l_bboxes, width=image.shape[1], height=image.shape[0])
            )
        return l_results
