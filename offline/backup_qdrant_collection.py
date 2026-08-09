#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable


@dataclass(frozen=True)
class SnapshotArtifact:
    path: Path
    sha256: str
    manifest: Path


Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]


def _default_transport(
    method: str, url: str, headers: dict[str, str], data: bytes | None = None
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.status, response.read()


def create_and_download_snapshot(
    qdrant_url: str,
    collection_name: str,
    backup_root: Path,
    bundle_version: str,
    api_key: str | None = None,
    transport: Transport | None = None,
    timestamp: str | None = None,
) -> SnapshotArtifact:
    call = transport or _default_transport
    headers = {"Accept": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    base = qdrant_url.rstrip("/")
    status, body = call(
        "POST", f"{base}/collections/{collection_name}/snapshots", headers, b""
    )
    if status // 100 != 2:
        raise RuntimeError(f"snapshot creation failed with HTTP {status}")
    snapshot_name = json.loads(body)["result"]["name"]
    status, info_body = call(
        "GET", f"{base}/collections/{collection_name}", headers, None
    )
    if status // 100 != 2:
        raise RuntimeError(f"collection info failed with HTTP {status}")
    info = json.loads(info_body)
    status, snapshot_bytes = call(
        "GET",
        f"{base}/collections/{collection_name}/snapshots/{snapshot_name}",
        headers,
        None,
    )
    if status // 100 != 2:
        raise RuntimeError(f"snapshot download failed with HTTP {status}")
    timestamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_root / collection_name / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    snapshot_path = destination / snapshot_name
    snapshot_path.write_bytes(snapshot_bytes)
    digest = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_path.with_name(snapshot_path.name + ".sha256").write_text(digest + "\n")
    manifest_path = destination / "backup-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collection": collection_name,
                "points": info["result"]["points_count"],
                "qdrant_version": info.get("version"),
                "snapshot_name": snapshot_name,
                "snapshot_sha256": digest,
                "bundle_version": bundle_version,
                "created_at": timestamp,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return SnapshotArtifact(snapshot_path, digest, manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="AIC_data_aic25_b1")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--bundle-version", default="1")
    parser.add_argument("--api-key")
    args = parser.parse_args()
    artifact = create_and_download_snapshot(
        args.qdrant_url,
        args.collection,
        args.destination,
        args.bundle_version,
        args.api_key,
    )
    print(json.dumps({"path": str(artifact.path), "sha256": artifact.sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
