from plate_formats import is_valid_plate as valid_format


def normalize_plate(number):
    translation = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
    return "".join(number.upper().translate(translation).split()).replace(
        "-", ""
    )


def is_valid_plate(number, subtype=None):
    number = normalize_plate(number)
    # Legacy one-argument clients retain complete civilian validation.
    if subtype is None:
        return "#" not in number and valid_format(number, "type1")
    return valid_format(number, subtype)


def confirmed_confidence(history, number):
    """No confidence when the confirmed label has left the rolling history."""
    matching = [
        item["confidence"] for item in history if item["plate"] == number
    ]
    return sum(matching) / len(matching) if matching else None


def plate_position(box, width, height):
    """Center offsets: pixels, positive right/down; x/y remain normalized."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    x, y = center_x / width, center_y / height
    return {
        "x": round(x - 0.5, 3),
        "y": round(y - 0.5, 3),
        "offset_x_px": round(center_x - width / 2),
        "offset_y_px": round(center_y - height / 2),
        "direction": "LEFT" if x < 0.4 else "RIGHT" if x > 0.6 else "CENTER",
    }
