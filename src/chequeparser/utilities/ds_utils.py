from typing import List
from chequeparser.datasets.base import BaseDS


def concat_ds(l_ds: List[BaseDS]) -> BaseDS:
    """Concatenates a list of datasets
    Assumes that all datasets are of same type
    Assumes that all datasets stem from same parent_ds
    """
    if l_ds is None or len(l_ds) == 0:
        return None
    if len(l_ds) == 1:
        return l_ds[0]
    return l_ds[0].__class__(
        raw=[item for ds in l_ds for item in ds.items],
        l_parent_idx=[item for ds in l_ds for item in ds.l_parent_idx],
        parent_ds=l_ds[0].parent_ds,
        batched=False,
        apply_gs=l_ds[0].apply_gs,
        size=l_ds[0].size,
    )
