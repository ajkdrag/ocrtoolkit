from typing import List, Optional

import numpy as np
from loguru import logger

from ocrtoolkit.utilities.model_utils import load_state_dict
from ocrtoolkit.wrappers.bbox import BBox
from ocrtoolkit.wrappers.detection_results import DetectionResults
from ocrtoolkit.wrappers.model import DetectionModel, RecognitionModel
from ocrtoolkit.wrappers.recognition_results import RecognitionResults

try:
    from doctr.models.predictor.base import _OCRPredictor
    from doctr.models.preprocessor import PreProcessor
except ImportError:
    logger.warning("Doctr is not installed.")


class DoctrDetModel(DetectionModel):

    def __init__(self, model, path, device, **kwargs):
        from doctr.models.detection.predictor import DetectionPredictor

        super().__init__(model, path, device)
        self.doctr_base_predictor = _OCRPredictor()
        kwargs["mean"] = kwargs.get("mean", model.cfg["mean"])
        kwargs["std"] = kwargs.get("std", model.cfg["std"])
        kwargs["batch_size"] = kwargs.get("batch_size", 2)
        input_shape = model.cfg["input_shape"][1:]
        self.predictor = DetectionPredictor(PreProcessor(input_shape, **kwargs), model)

    def _predict(self, images: List[np.ndarray], **kwargs) -> List[DetectionResults]:
        l_preds = self.predictor(images, **kwargs)
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


class DoctrRecModel(RecognitionModel):
    def __init__(self, model, path, device, **kwargs):
        from doctr.models.recognition.predictor import RecognitionPredictor

        super().__init__(model, path, device)
        kwargs.pop("pretrained_backbone", None)
        kwargs.pop("vocab", None)
        kwargs.pop("max_length", None)

        kwargs["mean"] = kwargs.get("mean", model.cfg["mean"])
        kwargs["std"] = kwargs.get("std", model.cfg["std"])
        kwargs["batch_size"] = kwargs.get("batch_size", 32)
        input_shape = model.cfg["input_shape"][-2:]
        self.predictor = RecognitionPredictor(
            PreProcessor(input_shape, preserve_aspect_ratio=True, **kwargs), model
        )

    def _predict(self, images: List[np.ndarray], **kwargs) -> List[RecognitionResults]:
        l_preds = self.predictor(images, **kwargs)
        l_results = []
        for image, preds in zip(images, l_preds):
            text, conf = preds
            l_results.append(
                RecognitionResults(
                    text=text, conf=conf, width=image.shape[1], height=image.shape[0]
                )
            )
        return l_results


def load(
    task: str,
    model_name: str,
    path: Optional[str],
    device: str,
    model_kwargs: dict,
    **kwargs,
):
    """Factory method to load model."""
    pretrained = model_kwargs.pop("pretrained", path is None or path == "")
    pretrained_backbone = model_kwargs.pop("pretrained_backbone", True)
    if path is None:
        path = "pretrained dir"

    if task == "det":
        from doctr.models import detection

        arch = detection.__dict__[model_name]
        model = arch(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            **model_kwargs,
        )
        if not pretrained:
            load_state_dict(path, model)
        return DoctrDetModel(model, path, device, **kwargs)
    elif task == "rec":
        from doctr.models import recognition

        arch = recognition.__dict__[model_name]
        model = arch(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            **model_kwargs,
        )
        if not pretrained:
            load_state_dict(path, model)
        return DoctrRecModel(model, path, device, **kwargs)
    else:
        raise NotImplementedError(f"Task {task} is not supported.")
