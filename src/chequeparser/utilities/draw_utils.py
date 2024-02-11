import cv2
import numpy as np


def draw_bbox(image: np.ndarray, 
              box: list, 
              label: str = "", 
              color: tuple = (255, 0, 0), 
              text_color: tuple = (255, 255, 255)):

    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, 
                  color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, 
                               fontScale=lw / 3, thickness=tf)[0] 
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, (*color, 1), -1, cv2.LINE_AA) 
        cv2.putText(image,
                    label, 
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    text_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
