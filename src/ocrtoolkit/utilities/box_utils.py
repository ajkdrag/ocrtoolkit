from typing import List, Tuple

import numpy as np

from ocrtoolkit.utilities.geometry_utils import estimate_page_angle, rotate_boxes


def sort_boxes(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort bounding boxes from top to bottom, left to right."""
    if boxes.ndim == 3:  # Rotated boxes
        angle = -estimate_page_angle(boxes)
        boxes = rotate_boxes(
            loc_preds=boxes, angle=angle, orig_shape=(1024, 1024), min_angle=5.0
        )
        boxes = np.concatenate((boxes.min(axis=1), boxes.max(axis=1)), axis=-1)
    sort_indices = (
        boxes[:, 0] + 2 * boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])
    ).argsort()
    return sort_indices, boxes


def resolve_sub_lines(
    boxes: np.ndarray, word_idcs: List[int], paragraph_break: float
) -> List[List[int]]:
    """Split a line in sub-lines."""
    lines = []
    word_idcs = sorted(word_idcs, key=lambda idx: boxes[idx, 0])

    if len(word_idcs) < 2:
        return [word_idcs]

    sub_line = [word_idcs[0]]
    for i in word_idcs[1:]:
        if boxes[i, 0] - boxes[sub_line[-1], 2] < paragraph_break:
            sub_line.append(i)
        else:
            lines.append(sub_line)
            sub_line = [i]
    lines.append(sub_line)
    return lines


def resolve_lines(boxes: np.ndarray, paragraph_break: float) -> List[List[int]]:
    """Order boxes to group them in lines."""
    idxs, boxes = sort_boxes(boxes)
    y_med = np.median(boxes[:, 3] - boxes[:, 1])

    lines, words, y_center_sum = [], [idxs[0]], boxes[idxs[0], [1, 3]].mean()
    for idx in idxs[1:]:
        y_dist = abs(boxes[idx, [1, 3]].mean() - y_center_sum / len(words))

        if y_dist < y_med / 2:
            words.append(idx)
            y_center_sum += boxes[idx, [1, 3]].mean()
        else:
            lines.extend(resolve_sub_lines(boxes, words, paragraph_break))
            words, y_center_sum = [idx], boxes[idx, [1, 3]].mean()

    if words:  # Process the last line
        lines.extend(resolve_sub_lines(boxes, words, paragraph_break))

    return lines
