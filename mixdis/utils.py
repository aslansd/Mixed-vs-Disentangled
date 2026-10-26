"""Small I/O helpers."""
import os, json
import numpy as np
import torch


def set_threads(n=None):
    torch.set_num_threads(n or max(1, (os.cpu_count() or 2) - 1))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, default=conv, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)
