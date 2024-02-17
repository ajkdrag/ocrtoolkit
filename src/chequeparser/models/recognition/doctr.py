from typing import Any, List

import numpy as np
from loguru import logger

from chequeparser.models.arch import BaseArch
from chequeparser.models.recognition.base import BaseRecognize
from chequeparser.wrappers.recognition_results import RecognitionResults


class DoctrRecognize(BaseRecognize):
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

    def predict_batch(
        self, images: List[np.ndarray], **kwargs
    ) -> List[RecognitionResults]:
        l_preds = self.network(images, **kwargs)
        l_results = []
        for image, preds in zip(images, l_preds):
            text, conf = preds
            l_results.append(RecognitionResults(text=text, conf=conf, np_img=image))
        return l_results
