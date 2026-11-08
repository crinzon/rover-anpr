"""Conservative representation of partially readable Russian license plates.

A question mark is allowed only when the upstream recognizer identifies the
position of an unreadable character. Never infer its position from a short OCR
string or silently substitute a plausible character.
"""
from dataclasses import dataclass
import re

LETTERS = "ABEKMHOPCTYX"
STANDARD = re.compile(rf"^[{LETTERS}]\d{{3}}[{LETTERS}]{{2}}\d{{2,3}}$")
PARTIAL = re.compile(rf"^[{LETTERS}?][\d?]{{3}}[{LETTERS}?]{{2}}[\d?]{{2,3}}$")


@dataclass(frozen=True)
class PlateReading:
    text: str | None
    status: str  # CONFIRMED, PARTIAL, UNREADABLE
    unknown_positions: tuple[int, ...]


def classify_plate(text: str | None, *, unknown_positions=None) -> PlateReading:
    """Classify without guessing. unknown_positions must come from the OCR pipeline.

    This function validates only the standard civilian layout. Other formats
    must be handled by explicit format-specific validators.
    """
    if not text:
        return PlateReading(None, "UNREADABLE", ())
    normalized = text.upper().replace(" ", "").replace("-", "")
    if STANDARD.fullmatch(normalized):
        return PlateReading(normalized, "CONFIRMED", ())
    if unknown_positions is None:
        return PlateReading(normalized, "UNREADABLE", ())
    positions = tuple(sorted(set(unknown_positions)))
    if (
        not positions
        or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(normalized) for i in positions)
        or any(normalized[i] != "?" for i in positions)
        or any(normalized[i] == "?" for i in range(len(normalized)) if i not in positions)
        or not PARTIAL.fullmatch(normalized)
    ):
        return PlateReading(normalized, "UNREADABLE", ())
    return PlateReading(normalized, "PARTIAL", positions)
