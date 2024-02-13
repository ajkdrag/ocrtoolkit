import numpy as np
from loguru import logger
from PIL import Image

from chequeparser.datasets.base import BaseDS
from chequeparser.models.recognition.base import BaseRecognize


def _recognize(model: BaseRecognize, ds: BaseDS, **kwargs):
    if not ds.batched:
        for idx, img in enumerate(ds):
            np_img = model.preprocess(np.array(img))
            recog_results = model.predict(np_img, **kwargs)
            recog_results.parent_ds = ds
            recog_results.parent_idx = idx
            yield recog_results
    else:
        l_np_imgs = [np.array(img) for img in ds]
        l_inputs = model.preprocess_batch(l_np_imgs)
        l_recog_results = model.predict_batch(l_inputs, **kwargs)
        for idx, det_results in enumerate(l_recog_results):
            recog_results.parent_ds = ds
            recog_results.parent_idx = idx
            yield res


def recognize(model: BaseRecognize, ds: BaseDS, stream=True, **kwargs):
    logger.info("Stream mode: {}", stream)
    logger.info("Batched mode: {}", ds.batched)
    logger.info("Running predict on {} samples", len(ds))
    gen = _recognize(model, ds, **kwargs)
    if stream:
        return gen
    return list(_recognize(model, ds, **kwargs))
