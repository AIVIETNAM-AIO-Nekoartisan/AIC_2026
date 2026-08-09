#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo


FIXED_TIMESTAMP = (2026, 8, 9, 0, 0, 0)


@dataclass(frozen=True)
class BundleArtifact:
    path: Path
    sha256: str
    size: int
    sidecar: Path
    map_videos: int
    map_keyframes: int


def inspect_map_archive(path: Path) -> tuple[int, int]:
    videos = 0
    rows = 0
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            videos += 1
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                required = {"n", "pts_time", "fps", "frame_idx"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"map CSV has invalid columns: {name}")
                rows += sum(1 for _ in reader)
    return videos, rows


def _approved_files(repo_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for source_root, target_root in (
        (repo_root / "colab" / "aic_colab", "aic_colab"),
        (repo_root / "colab" / "config", "config"),
        (repo_root / "colab" / "notebooks", "notebooks"),
        (repo_root / "tests" / "colab", "tests/colab"),
    ):
        for path in source_root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((path, f"{target_root}/{path.relative_to(source_root).as_posix()}"))
    for source, target in (
        (repo_root / "colab" / "requirements-colab.txt", "requirements-colab.txt"),
        (repo_root / "colab" / "README_COLAB.md", "README_COLAB.md"),
        (repo_root / "colab" / "VERSION", "VERSION"),
        (repo_root / "offline" / "import_colab_visual.py", "local_tools/import_colab_visual.py"),
        (repo_root / "offline" / "backup_qdrant_collection.py", "local_tools/backup_qdrant_collection.py"),
        (repo_root / "tests" / "test_colab_visual_import.py", "tests/test_colab_visual_import.py"),
        (
            repo_root / "docs" / "superpowers" / "specs" / "2026-08-09-colab-visual-encoding-bundle-design.md",
            "docs/DESIGN.md",
        ),
        (
            repo_root / "docs" / "superpowers" / "plans" / "2026-08-09-colab-visual-encoding-bundle.md",
            "docs/IMPLEMENTATION_PLAN.md",
        ),
    ):
        files.append((source, target))
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"bundle source files missing: {missing}")
    targets = [target for _, target in files]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate bundle target")
    return sorted(files, key=lambda item: item[1])


def _write_entry(archive: ZipFile, name: str, data: bytes, stored: bool = False) -> None:
    info = ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = ZIP_STORED if stored else ZIP_DEFLATED
    archive.writestr(info, data, compresslevel=None if stored else 9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle(repo_root: Path, map_archive: Path, output: Path) -> BundleArtifact:
    videos, rows = inspect_map_archive(map_archive)
    if (videos, rows) != (873, 177321):
        raise ValueError(
            f"map archive must contain exactly 873 CSV and 177321 rows; got {videos}/{rows}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w") as archive:
            for source, target in _approved_files(repo_root):
                _write_entry(archive, target, source.read_bytes())
            _write_entry(
                archive,
                "assets/map-keyframes-aic25-b1.zip",
                map_archive.read_bytes(),
                stored=True,
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = _sha256(output)
    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return BundleArtifact(
        output,
        digest,
        output.stat().st_size,
        sidecar,
        videos,
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    artifact = build_bundle(args.repo_root, args.map_archive, args.output)
    print(
        f"built {artifact.path} ({artifact.size} bytes)\n"
        f"map: {artifact.map_videos} videos / {artifact.map_keyframes} keyframes\n"
        f"sha256: {artifact.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
