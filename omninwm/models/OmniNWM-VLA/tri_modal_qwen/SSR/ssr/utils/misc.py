import os
import math
import pytz
import json
import shutup
import hashlib
import warnings
import numpy as np
import transformers
from torch import nn
from PIL import Image
from datetime import datetime
from typing import List, Dict, Any
from numerize.numerize import numerize


def quiet() -> None:
    shutup.please()
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    warnings.filterwarnings("ignore")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def change_ext(path: str, ext: str) -> str:
    return os.path.splitext(path)[0] + "." + ext


def colorize_depth(depth_path: str) -> np.ndarray:
    raw_depth = np.array(Image.open(depth_path))
    raw_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
    raw_depth = raw_depth.astype(np.uint8)
    colorized_depth = np.stack([raw_depth, raw_depth, raw_depth], axis=-1)
    return colorized_depth


def hash_str(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def count_params(model: nn.Module) -> str:
    total_params, tunable_params = 0, 0
    for param in model.parameters():
        n_params = param.numel()
        if n_params == 0 and hasattr(param, "ds_numel"):
            n_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            n_params *= 2
        total_params += n_params
        if param.requires_grad:
            tunable_params += n_params
    cnt_str = " || ".join(
        [f"Tunable Parameters: {numerize(tunable_params):<10}"
        , f"All Parameters: {numerize(total_params):<10}"
        , f"Tunable Ratio: {tunable_params / total_params:.5f}"
    ])
    return cnt_str


def str_datetime() -> str:
    return datetime.now(pytz.timezone("Asia/Shanghai")).strftime("[%Y-%m-%d %H:%M:%S,%f")[:-3] + "]"


def freeze_module(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def split_list(lst: List, n: int) -> List[List[Any]]:
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: List, n: int, k: int) -> List[Any]:
    chunks = split_list(lst, n)
    return chunks[k]


def accelerate_print(text: str, main_process: bool = True) -> None:
    if main_process:
        print(text)