import re

LETTERS = "ABEKMHOPCTYX"

PLATE_PATTERN = re.compile(rf"^[{LETTERS}]\d{{3}}[{LETTERS}]{{2}}\d{{2,3}}$")


def normalize_plate(number):
    return number.upper().replace(" ", "").replace("-", "")


def is_valid_plate(number):
    number = normalize_plate(number)
    return bool(PLATE_PATTERN.fullmatch(number))


def confirmed_confidence(history, number):
    """No confidence when the confirmed label has left the rolling history."""
    matching = [
        item["confidence"] for item in history if item["plate"] == number
    ]
    return sum(matching) / len(matching) if matching else None
