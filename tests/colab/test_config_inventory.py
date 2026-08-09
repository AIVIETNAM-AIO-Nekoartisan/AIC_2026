from pathlib import Path

import pytest

from aic_colab.config import load_config
from aic_colab.inventory import inspect_archives


def test_config_exposes_fixed_dataset_contract():
    cfg = load_config(Path("colab/config/aic25-b1.json"))

    assert cfg.dataset_id == "aic25-b1"
    assert len(cfg.archives) == 14
    assert cfg.siglip_model == "google/siglip2-so400m-patch14-384"
    assert cfg.siglip_dimension == 1152
    assert cfg.jina_model == "jinaai/jina-clip-v2"
    assert cfg.jina_dimension == 1024
    assert cfg.shard_size == 1024


def test_config_rejects_duplicate_archives(tmp_path):
    source = Path("colab/config/aic25-b1.json").read_text()
    source = source.replace('"Videos_L22_a.zip"', '"Videos_L21_a.zip"')
    path = tmp_path / "bad.json"
    path.write_text(source)

    with pytest.raises(ValueError, match="duplicate archive"):
        load_config(path)


def test_inventory_reports_missing_and_zero_byte_files(tmp_path):
    (tmp_path / "Videos_L21_a.zip").write_bytes(b"")

    report = inspect_archives(
        tmp_path,
        ("Videos_L21_a.zip", "Videos_L22_a.zip"),
        test_zip=False,
    )

    assert report.zero_byte == ("Videos_L21_a.zip",)
    assert report.missing == ("Videos_L22_a.zip",)
    assert not report.ready


def test_inventory_detects_corrupt_zip(tmp_path):
    (tmp_path / "Videos_L21_a.zip").write_bytes(b"not-a-zip")

    report = inspect_archives(
        tmp_path, ("Videos_L21_a.zip",), test_zip=True
    )

    assert report.corrupt == ("Videos_L21_a.zip",)
    assert not report.ready
