from typing import Dict, List, Tuple

import torch
from torch import Tensor

from plate_formats import (
    DIGITS as DIGITS,
)
from plate_formats import (
    DIPLOMATIC_LETTERS as DIPLOMATIC_LETTERS,
)
from plate_formats import (
    GOST_LETTERS as GOST_LETTERS,
)
from plate_formats import (
    LETTERS as LETTERS,
)
from plate_formats import (
    REGION3_PREFIX as REGION3_PREFIX,
)
from plate_formats import (
    REGION_START as REGION_START,
)
from plate_formats import (
    SLOT_CHARS as SLOT_CHARS,
)
from plate_formats import (
    SUBTYPE_MIN_LEN as SUBTYPE_MIN_LEN,
)
from plate_formats import (
    SUBTYPE_PATTERNS as SUBTYPE_PATTERNS,
)
from plate_formats import (
    SUBTYPES as SUBTYPES,
)
from plate_formats import (
    THREE_DIGIT_REGION_SUBTYPES as THREE_DIGIT_REGION_SUBTYPES,
)
from plate_formats import (
    UNKNOWN_SUBTYPE as UNKNOWN_SUBTYPE,
)
from plate_formats import (
    UNREADABLE_TOKEN as UNREADABLE_TOKEN,
)
from plate_formats import (
    is_valid_plate as is_valid_plate,
)

PAD_TOKEN: str = "<pad>"
BOS_TOKEN: str = "<bos>"
EOS_TOKEN: str = "<eos>"


alphabet: List[str] = (
    [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNREADABLE_TOKEN]
    + list(DIGITS)
    + list(LETTERS)
)

char_to_idx: Dict[str, int] = {char: idx for idx, char in enumerate(alphabet)}
idx_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(alphabet)}

PAD_IDX: int = char_to_idx[PAD_TOKEN]
BOS_IDX: int = char_to_idx[BOS_TOKEN]
EOS_IDX: int = char_to_idx[EOS_TOKEN]
UNREADABLE_IDX: int = char_to_idx[UNREADABLE_TOKEN]

num_classes: int = len(alphabet)

MAX_PLATE_LEN: int = 9
MAX_SEQ_LEN: int = MAX_PLATE_LEN + 1


subtype_to_idx: Dict[str, int] = {
    name: idx for idx, name in enumerate(SUBTYPES)
}
idx_to_subtype: Dict[int, str] = {
    idx: name for idx, name in enumerate(SUBTYPES)
}
num_subtypes: int = len(SUBTYPES)
UNKNOWN_SUBTYPE_IDX: int = subtype_to_idx[UNKNOWN_SUBTYPE]


_SLOT_IDS: Dict[str, List[int]] = {
    slot: [char_to_idx[c] for c in chars] for slot, chars in SLOT_CHARS.items()
}
_DIGIT_IDS: List[int] = [char_to_idx[c] for c in DIGITS]
REGION3_PREFIX_IDS: List[int] = [char_to_idx[c] for c in REGION3_PREFIX]

READABLE_SUBTYPE_IDS: Tuple[int, ...] = tuple(
    idx for name, idx in subtype_to_idx.items() if SUBTYPE_PATTERNS[name]
)


def is_readable_subtype(subtype: int) -> bool:
    return bool(SUBTYPE_PATTERNS[idx_to_subtype[subtype]])


def _pattern_mask(pattern: str, min_len: int) -> Tensor:
    masks: Tensor = torch.zeros(MAX_SEQ_LEN, num_classes, dtype=torch.bool)
    max_len: int = len(pattern)
    for pos in range(MAX_SEQ_LEN):
        if pos < max_len:
            masks[pos, _SLOT_IDS[pattern[pos]]] = True
            masks[pos, UNREADABLE_IDX] = True
        if min_len <= pos <= max_len:
            masks[pos, EOS_IDX] = True
    return masks


def build_position_masks() -> Tensor:
    return torch.stack(
        [
            _pattern_mask(SUBTYPE_PATTERNS[name], SUBTYPE_MIN_LEN[name])
            for name in SUBTYPES
        ]
    )


def build_union_position_mask() -> Tensor:
    return build_position_masks()[list(READABLE_SUBTYPE_IDS)].any(dim=0)


def build_region_gate() -> Tuple[Tensor, Tensor]:
    region_pos: Tensor = torch.full((num_subtypes,), -1, dtype=torch.long)
    max_len: Tensor = torch.zeros(num_subtypes, dtype=torch.long)
    for name, idx in subtype_to_idx.items():
        if name in REGION_START:
            region_pos[idx] = REGION_START[name]
            max_len[idx] = len(SUBTYPE_PATTERNS[name])
    return region_pos, max_len


def encode_plate(text: str) -> List[int]:
    return [char_to_idx[char] for char in text]


def decode_tokens(token_ids: Tensor) -> str:
    chars: List[str] = []
    for token in token_ids.tolist():
        if token in (EOS_IDX, PAD_IDX):
            break
        if token == BOS_IDX:
            continue
        chars.append(idx_to_char[token])
    return "".join(chars)


def infer_subtype(text: str, candidates: Tuple[str, ...] = SUBTYPES) -> int:
    for name in candidates:
        if SUBTYPE_PATTERNS[name] and is_valid_plate(text, name):
            return subtype_to_idx[name]
    return UNKNOWN_SUBTYPE_IDX
