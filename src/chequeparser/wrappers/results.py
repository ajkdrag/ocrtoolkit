import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from chequeparser.wrappers.bbox import BBox
from chequeparser.utilities.draw_utils import draw_bbox
from chequeparser.datasets.imageds import ImageDS


class DetectionResults:
    """Wrapper for detection results from a single image
    Captures the resulting bbox detections into a BBox object
    Stores the original input image width and height
    Methods to draw the bbox on a canvas
    """
    def __init__(self, bboxes: List[BBox], np_img: np.ndarray, 
                 parent_ds=None, parent_idx=None):
        self.bboxes = bboxes
        self.width = np_img.shape[1]
        self.height = np_img.shape[0]
        self.np_img = np_img
        self.parent_ds = parent_ds
        self.parent_idx = parent_idx

    def create_ds(self):
        """Creates an ImageDS from the crops of the DetectionResults"""
        crops = self.get_crops()
        image_ds = ImageDS(
                    crops,
                    size=None,
                    apply_gs=False,
                    batched=False,
                    parent_ds=self.parent_ds,
                    l_parent_idx=[self.parent_idx]*len(crops)
                )
        return image_ds

    def get_crops(self):
        """Returns a list of crops from the DetectionResults
        Takes each bbox and crops the image
        Returns a list of numpy arrays
        """
        return [
            self.np_img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            for bbox in self.bboxes
        ]
    
    def display(self, color: tuple=(255, 255, 255), 
                alpha=0.7, show_conf=True, show_label=True):
        """Displays the bboxes on a canvas
        If show_conf is True, displays confidence value 
        If show_label is True, displays label
        If boxes are normalized, it is denormalized before drawing
        If bbox color is dark, text color is light and vice versa
        """
        canvas = self.np_img.copy()
        black, white = (0, 0, 0), (255, 255, 255)
        text_color = white if np.mean(color) < 128 else black

        for bbox in self.bboxes:
            denorm_bbox = bbox

            if not bbox.normalized:
                denorm_bbox = bbox.denormalize(
                    self.width, self.height
                )
            
            str_label = "" 
            if show_label:
                str_label += f"{bbox.label}"
            if show_conf:
                str_label += f" {bbox.conf:.2f}"
            
            draw_bbox(canvas, denorm_bbox.values, str_label,
                      color=color, text_color=text_color)

        overlay = cv2.addWeighted(canvas, 
                      alpha, self.np_img, 1-alpha, gamma=0)
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)

