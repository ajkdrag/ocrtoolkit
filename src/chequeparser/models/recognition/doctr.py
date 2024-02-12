import numpy as np
from typing import Any, List
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
    predict_config: dict

    def __init__(self, arch, path=None, 
                 arch_config=None, predict_config=None):
        self.path = path
        self.arch = arch
        self.arch_config = arch_config if arch_config else {}
        self.predict_config = predict_config if predict_config else {}

        self.loader = arch(**self.arch_config)
        self.network = self.loader()

    def predict(self, 
                image: np.ndarray) -> RecognitionResults:
        return self.predict_batch([image])[0]
        
    def predict_batch(self, 
                      images: List[np.ndarray]) -> List[RecognitionResults]:
        l_preds = self.network(images, **self.predict_config)
        l_results = []
        for image, preds in zip(images, l_preds):
            text, conf = preds
            l_results.append(RecognitionResults(
                    text=text,
                    conf=conf,
                    np_img=image
                ))
        return l_results

