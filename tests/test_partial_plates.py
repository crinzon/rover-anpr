from partial_plates import classify_plate


def test_full_number():
    assert classify_plate("а123вс77").status == "CONFIRMED"


def test_partial_requires_explicit_unknown_position():
    assert classify_plate("A12?BC77").status == "UNREADABLE"
    result = classify_plate("A12?BC77", unknown_positions=[3])
    assert result.status == "PARTIAL"
    assert result.unknown_positions == (3,)


def test_does_not_invent_missing_character():
    assert classify_plate("A12BC77", unknown_positions=[3]).status == "UNREADABLE"


def test_bad_format_does_not_become_partial():
    assert classify_plate("ABC?1234", unknown_positions=[3]).status == "UNREADABLE"


def test_empty_reading():
    assert classify_plate(None).status == "UNREADABLE"
