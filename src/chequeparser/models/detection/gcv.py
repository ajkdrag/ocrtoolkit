from typing import List

import cv2
import numpy as np
from google.cloud import vision_v1 as vision

from chequeparser.models.detection.base import BaseDetect
from chequeparser.wrappers.detection_results import DetectionResults
from chequeparser.wrappers.textbbox import TextBBox


class GCVDetect(BaseDetect):
    svc_acc_path: str
    client: vision.ImageAnnotatorClient

    def __init__(self, svc_acc_path):
        self.svc_acc_path = svc_acc_path
        self.client = vision.ImageAnnotatorClient.from_service_account_file(
            svc_acc_path
        )

    def predict_batch(
        self, images: List[np.ndarray], **kwargs
    ) -> List[DetectionResults]:
        l_results = []
        for image in images:
            b64_img = cv2.imencode(".jpg", image)[1].tostring()
            vision_image = vision.types.Image(content=b64_img)
            response = self.client.text_detection(image=vision_image)
            l_bboxes = self.get_bounding_boxes(response)
            l_results.append(DetectionResults(l_bboxes, image))
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
                bbox = TextBBox(x1, y1, x2, y2, normalized=False)
                bbox.set_text_and_confidence(text, 1.0)
                bounding_boxes.append(bbox)

        return bounding_boxes
