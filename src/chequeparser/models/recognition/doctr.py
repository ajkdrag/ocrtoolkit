from typing import Any, List

import numpy as np

from chequeparser.models.arch import BaseArch
from chequeparser.models.recognition.base import BaseRecognize
from chequeparser.wrappers.recognition_results import RecognitionResults


class DoctrRecognize(BaseRecognize):
    path: str
    arch: BaseArch
    loader: Any
    network: Any
    arch_config: dict
    valid_kwargs = set()

    def __init__(self, arch, path=None, arch_config=None):
        self.path = path
        self.arch = arch
        self.arch_config = arch_config if arch_config else {}

        self.loader = arch(**self.arch_config)
        self.network = self.loader(path)

    def predict_batch(
        self, images: List[np.ndarray], **kwargs
    ) -> List[RecognitionResults]:
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in self.valid_kwargs
        }
        l_preds = self.network(images, **filtered_kwargs)
        l_results = []
        for image, preds in zip(images, l_preds):
            text, conf = preds
            l_results.append(
                RecognitionResults(
                    text=text, conf=conf, width=image.shape[1], height=image.shape[0]
                )
            )
        return l_results
