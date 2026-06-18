from src.utils.plate_normalizer import (
    choose_best_plate,
    is_valid_turkish_plate,
    normalize_plate_text,
)


def test_normalizes_plate_with_spaces() -> None:
    assert normalize_plate_text("34 ABC 123") == "34ABC123"


def test_normalizes_two_letter_plate() -> None:
    assert normalize_plate_text("06 AB 1234") == "06AB1234"


def test_normalizes_plate_with_separators() -> None:
    assert normalize_plate_text("34-ABC-123") == "34ABC123"


def test_rejects_invalid_province() -> None:
    assert normalize_plate_text("99ABC123") is None
    assert not is_valid_turkish_plate("99ABC123")


def test_empty_values_return_none() -> None:
    assert normalize_plate_text("") is None
    assert normalize_plate_text(None) is None


def test_choose_best_plate_prefers_valid_high_confidence_candidate() -> None:
    plate, confidence = choose_best_plate(
        [
            ("invalid", 0.99),
            ("34 ABC 123", 0.5),
            ("06 AB 1234", 0.7),
        ]
    )

    assert plate == "06AB1234"
    assert confidence == 0.7
