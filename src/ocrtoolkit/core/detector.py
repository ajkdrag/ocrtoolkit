import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from ocrtoolkit.datasets.base import BaseDS
from ocrtoolkit.utilities.det_utils import save_dets
from ocrtoolkit.wrappers.model import DetectionModel


def _detect(model: DetectionModel, ds: BaseDS, **kwargs):
    if not ds.batched:
        for idx, img in enumerate(ds):
            l_np_imgs = model.preprocess([np.array(img)])
            det_results = model.predict(l_np_imgs, **kwargs)[0]
            det_results.img_name = ds.names[idx]
            yield det_results
    else:
        l_np_imgs = [np.array(img) for img in ds]
        l_inputs = model.preprocess(l_np_imgs)
        l_det_results = model.predict(l_inputs, **kwargs)
        for idx, det_results in enumerate(l_det_results):
            det_results.img_name = ds.names[idx]
            yield det_results


def detect(model: DetectionModel, ds: BaseDS, stream=True, **kwargs):
    """Detects objects in a dataset
    Call model.preprocess methods before model.predict methods
    Images should be converted to np.ndarray before calling preprocess
    """
    if kwargs.get("verbose", True):
        logger.info("Stream mode: {}", stream)
        logger.info("Batched mode: {}", ds.batched)
        logger.info("Running predict on {} samples", len(ds))
    gen = _detect(model, ds, **kwargs)
    if stream:
        return gen
    return list(_detect(model, ds, **kwargs))


def detect_and_save_h5(
    model: DetectionModel,
    ds: BaseDS,
    path: str,
    bs=4,
    start_batch_idx=0,
    **kwargs,
):
    """Detects objects in a dataset
    Call model.preprocess methods before model.predict methods
    Images should be converted to np.ndarray before calling preprocess
    Saves the detections to path.
    Caution: This captures all detection results in mem before saving.
    """
    num_batches = ds.num_batches(bs)
    l_det_results = []
    for idx in tqdm(range(start_batch_idx, num_batches)):
        batch = ds.batch(bs, idx)
        l_det_results += detect(model, batch, stream=False, **kwargs)

    save_dets(l_det_results, path)
