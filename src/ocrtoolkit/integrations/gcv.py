from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

try:
    from google.cloud import vision_v1 as vision
    from google.oauth2 import service_account
except ImportError:
    logger.warning("Google Cloud Vision API is not installed.")
    pass

from ocrtoolkit.wrappers.bbox import BBox
from ocrtoolkit.wrappers.detection_results import DetectionResults
from ocrtoolkit.wrappers.model import DetectionModel


class GCVModel(DetectionModel):
    """GCVModel."""

    def __init__(self, client, path):
        super().__init__(model=client, path=path)

    def _predict(self, images: List[np.ndarray], **kwargs) -> List[DetectionResults]:
        l_results = []
        for image in images:
            b64_img = cv2.imencode(".jpg", image)[1].tostring()
            vision_image = vision.types.Image(content=b64_img)
            response = self.model.document_text_detection(image=vision_image)
            l_bboxes = self.get_bounding_boxes(response)
            l_results.append(
                DetectionResults(l_bboxes, width=image.shape[1], height=image.shape[0])
            )
        return l_results

    @staticmethod
    def get_bounding_boxes(response):
        bounding_boxes = []

        if response.text_annotations:
            for text_annotation in response.text_annotations[1:]:
                vertices = [
                    (vertex.x, vertex.y)
                    for vertex in text_annotation.bounding_poly.vertices
                ]

                vertices = np.array(vertices, dtype=np.int32)

                x1 = min(vertices[:, 0])
                y1 = min(vertices[:, 1])
                x2 = max(vertices[:, 0])
                y2 = max(vertices[:, 1])

                text = text_annotation.description
                bbox = BBox(x1, y1, x2, y2, normalized=False)
                bbox.set_text_and_confidence((text, 1.0))
                bounding_boxes.append(bbox)

        return bounding_boxes


def load(path: Optional[str], model_kwargs: dict, **kwargs):
    if path is not None:
        credentials = service_account.Credentials.from_service_account_file(path)
        model_kwargs["credentials"] = credentials

    client = vision.ImageAnnotatorClient(credentials=credentials)
    return GCVModel(client, path)
