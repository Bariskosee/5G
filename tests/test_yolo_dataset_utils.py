import pytest

from src.utils.yolo_dataset import YoloLabelError, parse_yolo_label_line


def test_parse_valid_yolo_label_line() -> None:
    annotation = parse_yolo_label_line(
        "0 0.500 0.250 0.100 0.200",
        class_count=1,
    )

    assert annotation.class_id == 0
    assert annotation.x_center == 0.5
    assert annotation.y_center == 0.25
    assert annotation.width == 0.1
    assert annotation.height == 0.2


def test_rejects_bbox_value_outside_unit_range() -> None:
    with pytest.raises(YoloLabelError, match="between 0 and 1"):
        parse_yolo_label_line("0 1.2 0.5 0.1 0.1", class_count=1)


def test_rejects_non_integer_class_id() -> None:
    with pytest.raises(YoloLabelError, match="class id must be an integer"):
        parse_yolo_label_line("0.5 0.5 0.5 0.1 0.1", class_count=2)


def test_rejects_class_id_out_of_range() -> None:
    with pytest.raises(YoloLabelError, match="outside valid range"):
        parse_yolo_label_line("2 0.5 0.5 0.1 0.1", class_count=2)
