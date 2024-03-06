def load_state_dict(path, model, ignore_keys: list = None):
    import torch

    state_dict = torch.load(path, map_location="cpu")
    if ignore_keys is not None and len(ignore_keys) > 0:
        for key in ignore_keys:
            state_dict.pop(key)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if set(missing_keys) != set(ignore_keys) or len(unexpected_keys) > 0:
            raise ValueError("Failed to load state_dict. Non-matching keys.")
    else:
        model.load_state_dict(state_dict)
