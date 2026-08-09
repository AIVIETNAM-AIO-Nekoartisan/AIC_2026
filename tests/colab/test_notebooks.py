import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name,command",
    [
        ("00_Setup_Preflight.ipynb", "preflight"),
        ("01_Pilot_1000_Frames.ipynb", "pilot"),
        ("02_Encode_Full_Dataset.ipynb", "full"),
    ],
)
def test_notebook_is_valid_thin_runner(name, command):
    document = json.loads((Path("colab/notebooks") / name).read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )

    assert document["nbformat"] == 4
    assert "drive.mount('/content/drive')" in source
    assert f"python -m aic_colab {command}" in source
    assert "DRIVE_ROOT" in source
    assert "BUNDLE_ROOT" in source
    assert len(document["cells"]) <= 7
    assert "GOOGLE_API_KEY" not in source
    assert "HF_TOKEN" not in source


def test_requirements_pin_transformers_below_five_and_cache_codec():
    requirements = Path("colab/requirements-colab.txt").read_text().splitlines()

    assert "transformers>=4.55,<5" in requirements
    assert "timm==0.6.13" in requirements
    assert "zstandard>=0.23,<1" in requirements
