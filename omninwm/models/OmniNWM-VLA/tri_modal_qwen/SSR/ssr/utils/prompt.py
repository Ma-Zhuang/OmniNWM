import re
from typing import List
from enum import Enum
from transformers import PreTrainedTokenizer


IGNORE_INDEX = -100


# 使用普通的Enum以兼容Python 3.9
class SSRSpecialToken(str, Enum):
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    BOR_TOKEN = "<|im_start|>"
    EOR_TOKEN = "<|im_end|>"
    TOR_TOKEN = "<tor>"
    IMAGE_TOKEN = "<image>"
    DEPTH_TOKEN = "<depth>"


def custom_join(lst: List[str]) -> str:
    if not lst:
        return ""
    it = iter(lst)
    result = [next(it)]
    prev_is_tor = (result[0] == SSRSpecialToken.TOR_TOKEN)
    for elem in it:
        if prev_is_tor or elem == SSRSpecialToken.TOR_TOKEN:
            result.append(elem)
        else:
            result.append(" " + elem)
        prev_is_tor = (elem == SSRSpecialToken.TOR_TOKEN)
    return "".join(result)


def insert_tor(sentence: str, n_tor: int) -> str:
    parts = sentence.split()
    if n_tor == 0:
        return sentence
    positions = []
    if len(parts) + 1 == 0:
        return " ".join([SSRSpecialToken.TOR_TOKEN] * n_tor)
    elif n_tor == 1:
        pos = round((len(parts) + 1 - 1) / 2)
        positions.append(pos)
    else:
        for i in range(n_tor):
            pos = round(i * (len(parts) + 1 - 1) / (n_tor - 1))
            positions.append(pos)
    count = {}
    for pos in positions:
        count[pos] = count.get(pos, 0) + 1
    new_elements = []
    for i in range(len(parts) + 1):
        new_elements.extend([SSRSpecialToken.TOR_TOKEN] * count.get(i, 0))
        if i < len(parts):
            new_elements.append(parts[i])
    if not new_elements and sentence.strip() == "" and n_tor > 0:
        new_elements = [SSRSpecialToken.TOR_TOKEN] * n_tor
    return custom_join(new_elements)


def string_truncation(
    text: str
    , tokenizer: PreTrainedTokenizer
    , max_length: int
) -> str:
    return tokenizer.decode(
        tokenizer.encode(
            text
            , max_length=max_length
            , truncation=True
            # , padding="max_length"
            , return_tensors="pt"
        )[0]
        , skip_special_tokens=True
    )