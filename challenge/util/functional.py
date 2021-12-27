from typing import List

import json
import operator
import functools
from functools import wraps

from datetime import datetime

import numpy as np
import torch
import torch.nn.utils.rnn as rnn


#
#
# -------- time_track -----------
#
def time_track(func):
    @wraps(func)
    def wrap(*args, **kw):
        t_begin = datetime.now()
        result = func(*args, **kw)
        t_end = datetime.now()

        print(
            f"[--- TIMETRACK || method: {func.__name__} -- time: {t_end - t_begin} ---]"
        )

        return result

    return wrap


#
#
#  -------- get_device -----------
#
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


#
#
#  -------- unpack -----------
#
def unpack(pack: rnn.PackedSequence) -> List[torch.Tensor]:
    """Convert the given packaged sequence into a list of vectors."""
    padded_pack, padded_len = rnn.pad_packed_sequence(
        pack, batch_first=True
    )
    return unpad(padded_pack, padded_len)


#
#
#  -------- unpad -----------
#
def unpad(padded: torch.Tensor, length: torch.Tensor) -> List[torch.Tensor]:
    """Convert the given packaged sequence into a list of vectors."""
    output = []
    for v, n in zip(padded, length):
        output.append(v[:n])
    return output


#
#
#  -------- flatten -----------
#
def flatten(lst: list):
    return functools.reduce(operator.iconcat, lst, [])


#
#
#  -------- load_json -----------
#
def load_json(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path) as data:
        return json.load(data)


#
#
#  -------- inverse_sigmoid -----------
#
def inverse_sigmoid(x) -> float:
    return 1 - (0.5 * (1 + np.sin((x * np.pi) - (np.pi / 2))))


#
#
#  -------- inverse_logistic -----------
#
def inverse_logistic(x, grow_rate: int = 3) -> float:
    return 1 / (1 + (x / (1 - x)) ** grow_rate)


#
#
#  -------- dict_max -----------
#
def dict_max(d: dict):
    return max(d.items(), key=operator.itemgetter(1))


#
#
#  -------- dict_min -----------
#
def dict_min(d: dict):
    return min(d.items(), key=operator.itemgetter(1))


#
#
#  -------- smooth_gradient -----------
#
def smooth_gradient(tensor: torch.Tensor, clip: float = 60.0):
    # possible problem: vanishing gradient
    # solution: set nan tensors to zero
    # src: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918
    tensor[tensor != tensor] = 0.0

    # possible problem: exploding gradient
    # solution: clamp all into the range [ min, max ]
    # src: https://pytorch.org/docs/stable/generated/torch.clamp.html
    torch.clamp(tensor, min=-clip, max=clip)

    return tensor
