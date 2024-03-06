import numpy as np
from typing import Optional, Tuple


def estimate_page_angle(polys: np.ndarray) -> float:
    """Takes a batch of rotated previously
    ORIENTED polys (N, 4, 2) (rectified by the classifier) and return the
    estimated angle ccw in degrees
    """
    # Compute mean left points and mean right point
    # with respect to the reading direction (oriented polygon)
    xleft = polys[:, 0, 0] + polys[:, 3, 0]
    yleft = polys[:, 0, 1] + polys[:, 3, 1]
    xright = polys[:, 1, 0] + polys[:, 2, 0]
    yright = polys[:, 1, 1] + polys[:, 2, 1]
    with np.errstate(divide="raise", invalid="raise"):
        try:
            return float(
                np.median(
                    np.arctan((yleft - yright) / (xright - xleft)) * 180 / np.pi
                )  # Y axis from top to bottom!
            )
        except FloatingPointError:
            return 0.0


def remap_boxes(
    loc_preds: np.ndarray, orig_shape: Tuple[int, int], dest_shape: Tuple[int, int]
) -> np.ndarray:
    """Remaps a batch of rotated locpred (N, 4, 2)
    expressed for an origin_shape to a destination_shape.
    This does not impact the absolute shape of the boxes,
    but allow to calculate the new relative RotatedBbox
    coordinates after a resizing of the image.

    Args:
    ----
        loc_preds: (N, 4, 2) array of RELATIVE loc_preds
        orig_shape: shape of the origin image
        dest_shape: shape of the destination image

    Returns:
    -------
        A batch of rotated loc_preds (N, 4, 2) expressed in the destination referencial
    """
    if len(dest_shape) != 2:
        raise ValueError(f"Mask length should be 2, was found at: {len(dest_shape)}")
    if len(orig_shape) != 2:
        raise ValueError(
            f"Image_shape length should be 2, was found at: {len(orig_shape)}"
        )
    orig_height, orig_width = orig_shape
    dest_height, dest_width = dest_shape
    mboxes = loc_preds.copy()
    mboxes[:, :, 0] = (
        (loc_preds[:, :, 0] * orig_width) + (dest_width - orig_width) / 2
    ) / dest_width
    mboxes[:, :, 1] = (
        (loc_preds[:, :, 1] * orig_height) + (dest_height - orig_height) / 2
    ) / dest_height

    return mboxes


def rotate_boxes(
    loc_preds: np.ndarray,
    angle: float,
    orig_shape: Tuple[int, int],
    min_angle: float = 1.0,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax, c)
    or rotated bounding boxes
    (4, 2) of an angle, if angle > min_angle, around the center of the page.
    If target_shape is specified, the boxes are
    remapped to the target shape after the rotation. This
    is done to remove the padding that is created by rotate_page(expand=True)

    Args:
    ----
        loc_preds: (N, 5) or (N, 4, 2) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        orig_shape: shape of the origin image
        min_angle: minimum angle to rotate boxes
        target_shape: shape of the destination image

    Returns:
    -------
        A batch of rotated boxes (N, 4, 2): or a batch of straight bounding boxes
    """
    # Change format of the boxes to rotated boxes
    _boxes = loc_preds.copy()
    if _boxes.ndim == 2:
        _boxes = np.stack(
            [
                _boxes[:, [0, 1]],
                _boxes[:, [2, 1]],
                _boxes[:, [2, 3]],
                _boxes[:, [0, 3]],
            ],
            axis=1,
        )
    # If small angle, return boxes (no rotation)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return _boxes
    # Compute rotation matrix
    angle_rad = angle * np.pi / 180.0  # compute radian angle for np functions
    rotation_mat = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=_boxes.dtype,
    )
    # Rotate absolute points
    points: np.ndarray = np.stack(
        (_boxes[:, :, 0] * orig_shape[1], _boxes[:, :, 1] * orig_shape[0]), axis=-1
    )
    image_center = (orig_shape[1] / 2, orig_shape[0] / 2)
    rotated_points = image_center + np.matmul(points - image_center, rotation_mat)
    rotated_boxes: np.ndarray = np.stack(
        (
            rotated_points[:, :, 0] / orig_shape[1],
            rotated_points[:, :, 1] / orig_shape[0],
        ),
        axis=-1,
    )

    # Apply a mask if requested
    if target_shape is not None:
        rotated_boxes = remap_boxes(
            rotated_boxes, orig_shape=orig_shape, dest_shape=target_shape
        )

    return rotated_boxes
