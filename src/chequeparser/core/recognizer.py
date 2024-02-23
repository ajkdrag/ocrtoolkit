import numpy as np
from loguru import logger

from chequeparser.datasets.base import BaseDS
from chequeparser.models.recognition import BaseRecognize


def _recognize(model: BaseRecognize, ds: BaseDS, **kwargs):
    if not ds.batched:
        for idx, img in enumerate(ds):
            np_img = model.preprocess(np.array(img))
            recog_results = model.predict(np_img, **kwargs)
            recog_results.img_name = ds.names[idx]
            yield recog_results
    else:
        l_np_imgs = [np.array(img) for img in ds]
        l_inputs = model.preprocess_batch(l_np_imgs)
        l_recog_results = model.predict_batch(l_inputs, **kwargs)
        for idx, recog_results in enumerate(l_recog_results):
            recog_results.img_name = ds.names[idx]
            yield recog_results


def recognize(model: BaseRecognize, ds: BaseDS, stream=True, **kwargs):
    if kwargs.get("verbose", True):
        logger.info("Stream mode: {}", stream)
        logger.info("Batched mode: {}", ds.batched)
        logger.info("Running predict on {} samples", len(ds))
    gen = _recognize(model, ds, **kwargs)
    if stream:
        return gen
    return list(_recognize(model, ds, **kwargs))
