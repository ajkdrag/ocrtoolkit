import numpy as np
from loguru import logger
from PIL import Image
from tqdm.autonotebook import tqdm

from chequeparser.datasets.base import BaseDS
from chequeparser.models.detection.base import BaseDetect


def _detect(model: BaseDetect, ds: BaseDS, **kwargs):
    if not ds.batched:
        for idx, img in enumerate(ds):
            np_img = model.preprocess(np.array(img))
            det_results = model.predict(np_img, **kwargs)
            det_results.img_name = ds.names[idx]
            yield det_results
    else:
        l_np_imgs = [np.array(img) for img in ds]
        l_inputs = model.preprocess_batch(l_np_imgs)
        l_det_results = model.predict_batch(l_inputs, **kwargs)
        for idx, det_results in enumerate(l_det_results):
            det_results.img_name = ds.names[idx]
            yield det_results


def detect(model: BaseDetect, ds: BaseDS, stream=True, **kwargs):
    """Detects objects in a dataset
    If ds is not batched, call model.predict(...)
    If ds is batched, call model.predict_batch(...)
    Call model.preprocess methods before model.predict methods
    Images should be converted to np.ndarray before calling preprocess
    """
    # TODO: Use a single predict method for batched and non-batched
    if kwargs.get("verbose", True):
        logger.info("Stream mode: {}", stream)
        logger.info("Batched mode: {}", ds.batched)
        logger.info("Running predict on {} samples", len(ds))
    gen = _detect(model, ds, **kwargs)
    if stream:
        return gen
    return list(_detect(model, ds, **kwargs))
