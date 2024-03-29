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


def reparameterize(model):
    import torch

    last_conv = None
    last_conv_name = None

    for module in model.modules():
        if hasattr(module, "reparameterize_layer"):
            module.reparameterize_layer()

    for name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            # fuse batchnorm only if it is followed by a conv layer
            if last_conv is None:
                continue
            conv_w = last_conv.weight
            conv_b = (
                last_conv.bias
                if last_conv.bias is not None
                else torch.zeros_like(child.running_mean)
            )

            factor = child.weight / torch.sqrt(child.running_var + child.eps)
            last_conv.weight = torch.nn.Parameter(
                conv_w * factor.reshape([last_conv.out_channels, 1, 1, 1])
            )
            last_conv.bias = torch.nn.Parameter(
                (conv_b - child.running_mean) * factor + child.bias
            )
            model._modules[last_conv_name] = last_conv
            model._modules[name] = torch.nn.Identity()
            last_conv = None
        elif isinstance(child, torch.nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            reparameterize(child)

    return model
