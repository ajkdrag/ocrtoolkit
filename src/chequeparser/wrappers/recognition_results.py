import numpy as np
import matplotlib.pyplot as plt
from typing import List
from chequeparser.utilities.draw_utils import draw_ocr_text


class RecognitionResults:
    def __init__(self, text, conf, np_img: np.ndarray,
                 parent_ds=None, parent_idx=None):
        self.text = text
        self.conf = conf
        self.np_img = np_img
        self.parent_ds = parent_ds
        self.parent_idx = parent_idx

    def display(self, color: tuple=(255, 255, 255),
                show_conf=False):
        """Displays the text and confidence on a canvas"""
        canvas = self.np_img.copy()
        black, white = (0, 0, 0), (255, 255, 255)
        text_color = white if np.mean(color) < 128 else black

        str_label = f"{self.text}"
        if show_conf:
            str_label += f" [{self.conf:.2f}]"

        canvas = draw_ocr_text(canvas, str_label, color=color,
                      text_color=text_color)
        
        plt.figure(figsize=(6, 2))
        plt.axis("off")
        plt.imshow(canvas)

