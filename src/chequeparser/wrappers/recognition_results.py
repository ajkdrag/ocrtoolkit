from typing import List

import matplotlib.pyplot as plt
import numpy as np

from chequeparser.utilities.draw_utils import draw_ocr_text


class RecognitionResults:
    text: str
    conf: float
    width: int
    height: int
    img_name: str

    def __init__(
        self,
        text,
        conf,
        width,
        height,
        img_name,
    ):
        self.text = text
        self.conf = conf
        self.width = width
        self.height = height
        self.img_name = img_name

    def __len__(self):
        return len(self.text)

    def draw(
        self,
        parent_ds: "BaseDS",
        color: tuple = (255, 0, 0),
        show_conf=False,
        show_text=True,
        display=True,
    ) -> np.ndarray:
        """Draws the text and confidence on a canvas"""
        canvas = np.array(parent_ds[self.img_name])
        black, white = (0, 0, 0), (255, 255, 255)
        text_color = white if np.mean(color) < 128 else black

        str_label = f"{self.text}"
        if show_conf:
            str_label += f" [{self.conf:.2f}]"

        canvas = draw_ocr_text(canvas, str_label, color=color, text_color=text_color)

        if display:
            plt.figure(figsize=(6, 2))
            plt.axis("off")
            plt.imshow(canvas)
        else:
            return canvas
