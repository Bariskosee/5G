import json

from src.output.result_builder import build_default_result, build_result, write_results_json
from src.output.schema import PLATE_FALLBACK, clamp_confidence, validate_output_schema


def test_clamp_confidence() -> None:
    assert clamp_confidence(-1) == 0.0
    assert clamp_confidence(0.42) == 0.42
    assert clamp_confidence(2) == 1.0
    assert clamp_confidence("bad") == 0.0


def test_default_result_uses_official_schema_and_fallback_plate() -> None:
    result = build_default_result("video.mp4")

    assert set(result) == {"video_id", "arac_bilgisi", "tespitler"}
    assert result["arac_bilgisi"]["plaka"] == PLATE_FALLBACK
    assert result["arac_bilgisi"]["confidence_score"] == 0.01
    assert validate_output_schema(result) == []


def test_result_schema_accepts_official_detection_labels() -> None:
    result = build_result(
        video_id="video.mp4",
        vehicle_type="sedan",
        plate="34ABC123",
        color="beyaz",
        vehicle_confidence=0.86,
        events=[
            {
                "zaman_saniye": 1.5,
                "kategori": "sofor_eylemi",
                "etiket": "telefonla_konusma",
                "confidence_score": 0.7,
            }
        ],
    )

    assert validate_output_schema(result) == []


def test_result_schema_rejects_invalid_category_label_pair() -> None:
    result = build_default_result("video.mp4")
    result["tespitler"].append(
        {
            "zaman_saniye": 1.5,
            "kategori": "nesneler",
            "etiket": "telefonla_konusma",
            "confidence_score": 0.7,
        }
    )

    errors = validate_output_schema(result)
    assert any("etiket" in error for error in errors)


def test_result_json_can_be_written_and_parsed(tmp_path) -> None:
    result = build_default_result("video.mp4")
    output_path = tmp_path / "results.json"

    write_results_json(result, output_path)
    parsed = json.loads(output_path.read_text(encoding="utf-8"))

    assert parsed == result
    assert validate_output_schema(parsed) == []
