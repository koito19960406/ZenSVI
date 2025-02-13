# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch


def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel.

    This function checks for the "model" key in the state_dict. DataParallel prefixes
    state_dict keys with 'module.' when saving. If the model is not a DataParallel model
    but the state_dict is, then prefixes are removed. If the model is a DataParallel model
    but the state_dict is not, then prefixes are added.

    Args:
        model (torch.nn.Module): The model to load the state_dict into.
        state_dict (dict): The state dictionary containing model weights.

    Returns:
        torch.nn.Module: The model with loaded state_dict.
    """
    state_dict = state_dict.get("model", state_dict)

    do_prefix = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith("module.") and not do_prefix:
            k = k[7:]

        if not k.startswith("module.") and do_prefix:
            k = "module." + k

        state[k] = v

    model.load_state_dict(state)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    """Load weights from a checkpoint file into the model.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return load_state_dict(model, ckpt)


def load_state_dict_from_url(model, url, **kwargs):
    """Load state_dict from a URL into the model.

    Args:
        model (torch.nn.Module): The model to load the state_dict into.
        url (str): The URL to load the state_dict from.
        **kwargs: Additional arguments for loading the state_dict.

    Returns:
        torch.nn.Module: The model with loaded state_dict.
    """
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", **kwargs)
    return load_state_dict(model, state_dict)


def load_state_from_resource(model, resource: str):
    """Loads weights to the model from a given resource.

    A resource can be of the following types:
        1. URL. Prefixed with "url::"
           e.g. url::http(s)://url.resource.com/ckpt.pt
        2. Local path. Prefixed with "local::"
           e.g. local::/path/to/ckpt.pt

    Args:
        model (torch.nn.Module): The model to load weights into.
        resource (str): The resource string indicating the source of weights.

    Returns:
        torch.nn.Module: The model with loaded weights.

    Raises:
        ValueError: If the resource type is invalid.
    """
    print(f"Using pretrained resource {resource}")

    if resource.startswith("url::"):
        url = resource.split("url::")[1]
        return load_state_dict_from_url(model, url, progress=True)

    elif resource.startswith("local::"):
        path = resource.split("local::")[1]
        return load_wts(model, path)

    else:
        raise ValueError("Invalid resource type, only url:: and local:: are supported")
