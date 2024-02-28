def concat_ds(l_ds, reset_names=False):
    """Concatenates a list of datasets
    Assumes that all datasets are of same type
    Assumes that all datasets stem from same parent_ds
    """
    if l_ds is None or len(l_ds) == 0:
        return None
    if len(l_ds) == 1:
        return l_ds[0]

    concatenated_ds = l_ds[0].__class__(
        items=[item for ds in l_ds for item in ds.items],
        names=[name for ds in l_ds for name in ds.names],
        batched=False,
        apply_gs=l_ds[0].apply_gs,
        size=l_ds[0].size,
    )

    if reset_names:
        concatenated_ds._setup_names()
    return concatenated_ds
