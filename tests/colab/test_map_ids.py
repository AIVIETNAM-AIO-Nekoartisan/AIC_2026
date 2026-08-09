from pathlib import Path

import pytest

from aic_colab.ids import stable_point_id
from aic_colab.map_reader import read_map_csv


def test_stable_point_id_matches_approved_design_example():
    assert (
        stable_point_id("aic25-b1", "L21_V001", 1)
        == "9c6e7432-1908-54ea-8c96-8bbff30a1fc2"
    )


def test_map_reader_converts_rows_to_records(tmp_path):
    path = tmp_path / "L21_V001.csv"
    path.write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,0.0,25,0\n"
        "2,1.0,25,25\n"
    )

    rows = read_map_csv(path, "aic25-b1")

    assert [(r.video_id, r.ordinal, r.frame_index) for r in rows] == [
        ("L21_V001", 1, 0),
        ("L21_V001", 2, 25),
    ]
    assert rows[0].point_id == "9c6e7432-1908-54ea-8c96-8bbff30a1fc2"


def test_map_reader_rejects_non_monotonic_rows(tmp_path):
    path = tmp_path / "L21_V001.csv"
    path.write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,1.0,25,25\n"
        "2,0.0,25,0\n"
    )

    with pytest.raises(ValueError, match="monotonic"):
        read_map_csv(path, "aic25-b1")


@pytest.mark.parametrize(
    "header",
    [
        "n,pts_time,fps\n1,0,25\n",
        "n,pts_time,frame_idx\n1,0,0\n",
    ],
)
def test_map_reader_rejects_missing_required_columns(tmp_path, header):
    path = tmp_path / "L21_V001.csv"
    path.write_text(header)

    with pytest.raises(ValueError, match="required columns"):
        read_map_csv(path, "aic25-b1")
