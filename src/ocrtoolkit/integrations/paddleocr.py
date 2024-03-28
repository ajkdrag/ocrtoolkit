from typing import List, Optional

import importlib_resources
import numpy as np
from loguru import logger

try:
    from paddleocr.tools.infer import utility
    from ppocr import utils as pputils

    EN_DICT_FILE = importlib_resources.files(pputils).joinpath("en_dict.txt").as_posix()
except ImportError:
    EN_DICT_FILE = None
    logger.warning("PaddleOCR not installed")


from ocrtoolkit.utilities.network_utils import download_file
from ocrtoolkit.wrappers.bbox import BBox
from ocrtoolkit.wrappers.detection_results import DetectionResults
from ocrtoolkit.wrappers.model import DetectionModel, RecognitionModel
from ocrtoolkit.wrappers.recognition_results import RecognitionResults

BASE_URL = "https://paddleocr.bj.bcebos.com"
DET_MODEL_URLS = {
    "DB": f"{BASE_URL}/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    "SVTR_LCNet": f"{BASE_URL}/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
}


class PaddleOCRDetModel(DetectionModel):
    def _predict(self, images: List[np.ndarray], **kwargs) -> List[DetectionResults]:
        import torch

        l_results = []

        for image in images:
            with torch.inference_mode():
                preds, _ = self.model(image, **kwargs)
            logger.info(preds.shape)
            l_bboxes = self.get_bounding_boxes(preds)
            l_results.append(
                DetectionResults(l_bboxes, width=image.shape[1], height=image.shape[0])
            )
        return l_results

    @staticmethod
    def get_bounding_boxes(dt_boxes):
        """Returns BBox objects from numpy polygon results
        dt_boxes shape: nx4x2
        Use only numpy operations and no for loops
        return: list of BBox
        """
        dt_boxes_reshaped = dt_boxes.reshape(-1, 2, 4)
        x_min = np.min(dt_boxes_reshaped[:, :, 0], axis=1)
        y_min = np.min(dt_boxes_reshaped[:, :, 1], axis=1)
        x_max = np.max(dt_boxes_reshaped[:, :, 0], axis=1)
        y_max = np.max(dt_boxes_reshaped[:, :, 1], axis=1)

        xyxy_boxes = np.stack((x_min, y_min, x_max, y_max), axis=1)

        return [BBox.from_numpy(bbox) for bbox in xyxy_boxes]


class PaddleOCRRecModel(RecognitionModel):
    def _predict(self, images: List[np.ndarray], **kwargs) -> List[RecognitionResults]:
        import torch

        with torch.inference_mode():
            l_preds, _ = self.model(images, **kwargs)
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
    if path is None:
        try:
            path = download_file(
                DET_MODEL_URLS.get(model_name), cache_subdir="ppocr/models/", untar=True
            )
        except Exception as e:
            raise ValueError(f"No pretrained model for {model_name}.") from e

    model_kwargs["use_gpu"] = device != "cpu"
    model_kwargs["rec_char_dict_path"] = model_kwargs.get(
        "rec_char_dict_path", EN_DICT_FILE
    )
    ppocr_parser = utility.init_args()
    all_args = ppocr_parser.parse_args("")
    dict_args = vars(all_args)
    dict_args.update(model_kwargs)

    if task == "det":
        from paddleocr.tools.infer import predict_det

        dict_args["det_model_dir"] = path
        logger.info(all_args)
        predictor = predict_det.TextDetector(all_args)
        return PaddleOCRDetModel(predictor, path, device, **kwargs)

    elif task == "rec":
        from paddleocr.tools.infer import predict_rec

        dict_args["rec_model_dir"] = path
        logger.info(all_args)
        predictor = predict_rec.TextRecognizer(all_args)
        return PaddleOCRRecModel(predictor, path, device, **kwargs)

    else:
        raise NotImplementedError(f"Task {task} is not supported.")
