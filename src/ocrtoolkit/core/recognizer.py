import numpy as np
from loguru import logger

from ocrtoolkit.datasets.base import BaseDS
from ocrtoolkit.wrappers.model import RecognitionModel


def _recognize(model: RecognitionModel, ds: BaseDS, **kwargs):
    if not ds.batched:
        for idx, img in enumerate(ds):
            l_np_imgs = model.preprocess([np.array(img)])
            recog_results = model.predict(l_np_imgs, **kwargs)[0]
            recog_results.img_name = ds.names[idx]
            yield recog_results
    else:
        l_np_imgs = [np.array(img) for img in ds]
        l_inputs = model.preprocess(l_np_imgs)
        l_recog_results = model.predict(l_inputs, **kwargs)
        for idx, recog_results in enumerate(l_recog_results):
            recog_results.img_name = ds.names[idx]
            yield recog_results


def recognize(model: RecognitionModel, ds: BaseDS, stream=True, **kwargs):
    """Recognizes text in a dataset
    Call model.preprocess methods before model.predict methods
    Images should be converted to np.ndarray before calling preprocess
    """
    if kwargs.get("verbose", True):
        logger.info("Stream mode: {}", stream)
        logger.info("Batched mode: {}", ds.batched)
        logger.info("Running predict on {} samples", len(ds))
    gen = _recognize(model, ds, **kwargs)
    if stream:
        return gen
    return list(_recognize(model, ds, **kwargs))
