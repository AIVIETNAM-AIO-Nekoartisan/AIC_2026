from pathlib import Path
from zipfile import ZipFile

import pytest

from build_bundle import build_bundle, inspect_map_archive


REPO = Path(__file__).resolve().parents[2]
MAP_ARCHIVE = Path("/media/nekoartisan/Lexar/AIC/videos/map-keyframes-aic25-b1.zip")


def test_real_map_archive_has_approved_inventory():
    assert inspect_map_archive(MAP_ARCHIVE) == (873, 177321)


def test_bundle_is_deterministic_and_excludes_runtime_data(tmp_path):
    one = build_bundle(REPO, MAP_ARCHIVE, tmp_path / "one.zip")
    two = build_bundle(REPO, MAP_ARCHIVE, tmp_path / "two.zip")

    assert one.sha256 == two.sha256
    with ZipFile(one.path) as archive:
        names = set(archive.namelist())
    assert "assets/map-keyframes-aic25-b1.zip" in names
    assert "local_tools/import_colab_visual.py" in names
    assert "local_tools/backup_qdrant_collection.py" in names
    assert "notebooks/00_Setup_Preflight.ipynb" in names
    assert "aic_colab/runner.py" in names
    assert not any(
        forbidden in name
        for name in names
        for forbidden in (".env", "qdrant_data", "es_data", "results/", ".runtime")
    )


def test_bundle_rejects_wrong_map_inventory(tmp_path):
    archive = tmp_path / "map.zip"
    with ZipFile(archive, "w") as output:
        output.writestr(
            "map-keyframes/L21_V001.csv",
            "n,pts_time,fps,frame_idx\n1,0,25,0\n",
        )

    with pytest.raises(ValueError, match="873 CSV and 177321 rows"):
        build_bundle(REPO, archive, tmp_path / "bad.zip")
