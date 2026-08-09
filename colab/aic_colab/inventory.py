from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile


@dataclass(frozen=True)
class ArchiveInventory:
    present: tuple[str, ...]
    missing: tuple[str, ...]
    zero_byte: tuple[str, ...]
    corrupt: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not (self.missing or self.zero_byte or self.corrupt)


def inspect_archives(
    root: Path, expected: tuple[str, ...], test_zip: bool = True
) -> ArchiveInventory:
    present: list[str] = []
    missing: list[str] = []
    zero_byte: list[str] = []
    corrupt: list[str] = []
    for name in sorted(expected):
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        present.append(name)
        if path.stat().st_size == 0:
            zero_byte.append(name)
            continue
        if test_zip:
            try:
                with ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        corrupt.append(name)
            except (BadZipFile, OSError):
                corrupt.append(name)
    return ArchiveInventory(
        present=tuple(present),
        missing=tuple(missing),
        zero_byte=tuple(zero_byte),
        corrupt=tuple(corrupt),
    )
