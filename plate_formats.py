"""Shared OCR format definitions. No neural/runtime dependencies."""

from typing import Dict, Tuple

UNREADABLE_TOKEN: str = "#"

DIGITS: str = "0123456789"

GOST_LETTERS: str = "ABEKMHOPCTYX"

DIPLOMATIC_LETTERS: str = "D"

LETTERS: str = GOST_LETTERS + DIPLOMATIC_LETTERS

UNKNOWN_SUBTYPE: str = "unknown"

SUBTYPES: Tuple[str, ...] = (
    "type1",
    "type1a",
    "type1b",
    "type2",
    "type3",
    "type4",
    "type4a",
    "type4b",
    "type5",
    "type6",
    "type7",
    "type8",
    "type9",
    "type10",
    "type11",
    "type15",
    "type16",
    "type17",
    "type18",
    "type19",
    "type20",
    "type21",
    "type22",
    "type23",
    "type24",
    "type25",
    "type26",
    "type27",
    "type28",
    UNKNOWN_SUBTYPE,
)

SLOT_CHARS: Dict[str, str] = {
    "L": GOST_LETTERS,
    "D": DIGITS,
    "s": "DT",
    "d": "D",
    "t": "T",
    "k": "K",
    "c": "C",
    "C": "C",
}

SUBTYPE_PATTERNS: Dict[str, str] = {
    "type1": "LDDDLLDDD",
    "type1a": "LDDDLLDDD",
    "type1b": "LLDDDDD",
    "type2": "LLDDDDDD",
    "type3": "DDDDLLDD",
    "type4": "DDDDLLDD",
    "type4a": "LLDDDDDD",
    "type4b": "LLDDLLDD",
    "type5": "DDDDLLDD",
    "type6": "LLDDDDDD",
    "type7": "DDDDLLDD",
    "type8": "DDDDLLDD",
    "type9": "DDDCdDDD",
    "type10": "DDDsDDDDD",
    "type11": "sDDDDDDD",
    "type15": "LLDDDLDD",
    "type16": "LLDDDDDD",
    "type17": "LLDDDDDD",
    "type18": "LLDDDDDD",
    "type19": "tLLDDDDD",
    "type20": "LDDDDDD",
    "type21": "DDDLDD",
    "type22": "DDDDLDD",
    "type23": "kLLDDDDD",
    "type24": "kLLDDDDD",
    "type25": "kDDDLLDD",
    "type26": "cLLDDDDD",
    "type27": "cLLDDDDD",
    "type28": "cDDDLLDD",
    UNKNOWN_SUBTYPE: "",
}

THREE_DIGIT_REGION_SUBTYPES: Tuple[str, ...] = ("type1", "type1a")

REGION_START: Dict[str, int] = {
    name: 6 for name in THREE_DIGIT_REGION_SUBTYPES
}

SUBTYPE_MIN_LEN: Dict[str, int] = {
    name: len(pattern) - (1 if name in THREE_DIGIT_REGION_SUBTYPES else 0)
    for name, pattern in SUBTYPE_PATTERNS.items()
}

REGION3_PREFIX: str = "123456789"


def is_valid_plate(text: str, subtype: str) -> bool:
    pattern: str = SUBTYPE_PATTERNS.get(subtype, "")
    if not pattern:
        return False
    if not SUBTYPE_MIN_LEN[subtype] <= len(text) <= len(pattern):
        return False
    for char, slot in zip(text, pattern):
        if char != UNREADABLE_TOKEN and char not in SLOT_CHARS[slot]:
            return False
    if len(text) == len(pattern) and subtype in REGION_START:
        start: int = REGION_START[subtype]
        if (
            text[start] not in REGION3_PREFIX
            and text[start] != UNREADABLE_TOKEN
        ):
            return False
    return True
