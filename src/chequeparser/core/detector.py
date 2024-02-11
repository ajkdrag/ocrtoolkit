import numpy as np
from loguru import logger
from PIL import Image
from chequeparser.models.detection.base import BaseDetect
from chequeparser.datasets.base import BaseDS


def detect(model: BaseDetect,
           ds: BaseDS, stream=True, **kwargs):
    """Detects objects in a dataset
    If ds is not batched, call model.predict(...)
    If ds is batched, call model.predict_batch(...)
    Call model.preprocess methods before model.predict methods
    Images should be converted to np.ndarray before calling preprocess
    """
    # TODO: Use a single predict method for batched and non-batched
    logger.info("Stream mode: {}", stream)
    logger.info("Batched mode: {}", ds.batched)
    logger.info("Running predict on {} samples", len(ds))
    gen = _detect(model, ds, **kwargs)
    if stream: return gen
    return list(_detect(model, ds, **kwargs))

    
def _detect(model: BaseDetect,
           ds: BaseDS, **kwargs):
    if not ds.batched:
        for img in ds:
            np_img = model.preprocess(np.array(img))
            yield model.predict(np_img, **kwargs)
    else:
        l_np_imgs = [np.array(img) for img in ds]
        l_inputs = model.preprocess_batch(l_np_imgs)
        l_results = model.predict_batch(l_inputs, **kwargs)
        for res in l_results:
            yield res

