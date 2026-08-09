#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class VisualShard:
    batch_id: str
    shard_id: str
    tar_path: Path
    manifest_path: Path
    siglip_path: Path
    jina_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"missing artifact or checksum: {path}")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    if _sha256(path) != expected:
        raise ValueError(f"checksum mismatch: {path}")


def discover_visual_shards(results_root: Path) -> list[VisualShard]:
    shards: list[VisualShard] = []
    for manifest in sorted((results_root / "manifests").glob("*/*.jsonl")):
        batch_id = manifest.parent.name
        shard_id = manifest.stem
        shard = VisualShard(
            batch_id=batch_id,
            shard_id=shard_id,
            tar_path=results_root / "keyframes" / batch_id / f"{shard_id}.tar",
            manifest_path=manifest,
            siglip_path=results_root / "embeddings" / batch_id / f"{shard_id}.siglip.npz",
            jina_path=results_root / "embeddings" / batch_id / f"{shard_id}.jina.npz",
        )
        for path in (
            shard.tar_path,
            shard.manifest_path,
            shard.siglip_path,
            shard.jina_path,
        ):
            _verify(path)
        shards.append(shard)
    if not shards:
        raise ValueError(f"no complete visual shards found under {results_root}")
    return shards


def verify_and_extract_keyframes(
    shards: Sequence[VisualShard], keyframe_root: Path
) -> int:
    extracted = 0
    keyframe_root.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        _verify(shard.tar_path)
        with tarfile.open(shard.tar_path) as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or member.issym()
                    or member.islnk()
                    or not member.isfile()
                ):
                    raise ValueError(f"unsafe tar member: {member.name}")
                target = keyframe_root.joinpath(*relative.parts)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read tar member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                extracted += 1
    return extracted


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_vectors(path: Path, expected_dim: int) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"point_ids", "vectors"}:
            raise ValueError(f"invalid NPZ schema: {path}")
        ids = archive["point_ids"].astype(str).tolist()
        vectors = archive["vectors"]
    if vectors.dtype != np.float32 or vectors.shape != (len(ids), expected_dim):
        raise ValueError(f"invalid vector shape/dtype: {path}")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"non-finite vector: {path}")
    return ids, vectors


def import_visual_shards(
    client,
    collection_name: str,
    shards: Sequence[VisualShard],
    keyframe_root: Path,
    batch_size: int = 256,
    resume_existing: bool = False,
) -> int:
    if client.collection_exists(collection_name):
        if not resume_existing:
            raise ValueError(f"collection already exists: {collection_name}")
        _validate_existing_schema(client, collection_name)
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "siglip": models.VectorParams(size=1152, distance=models.Distance.COSINE),
                "jina": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            },
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    imported = 0
    for shard in shards:
        rows = _read_manifest(shard.manifest_path)
        siglip_ids, siglip = _load_vectors(shard.siglip_path, 1152)
        jina_ids, jina = _load_vectors(shard.jina_path, 1024)
        manifest_ids = [str(row["point_id"]) for row in rows]
        if siglip_ids != manifest_ids or jina_ids != manifest_ids:
            raise ValueError(f"point ID mismatch in {shard.batch_id}/{shard.shard_id}")
        points = []
        for index, row in enumerate(rows):
            local_path = keyframe_root.joinpath(*PurePosixPath(row["keyframe_member"]).parts)
            if not local_path.is_file():
                raise ValueError(f"missing extracted keyframe: {local_path}")
            payload = {
                "dataset_id": row["dataset_id"],
                "video": {"name": row["video_id"]},
                "frame": {
                    "ordinal": row["ordinal"],
                    "frame_index": row["frame_index"],
                    "timestamp_seconds": row["timestamp_seconds"],
                    "path": str(local_path),
                },
            }
            points.append(
                models.PointStruct(
                    id=row["point_id"],
                    vector={"siglip": siglip[index].tolist(), "jina": jina[index].tolist()},
                    payload=payload,
                )
            )
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )
        imported += len(points)
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    for field, schema in (
        ("frame.timestamp_seconds", models.PayloadSchemaType.FLOAT),
        ("video.name", models.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema,
            wait=True,
        )
    return imported


def _validate_existing_schema(client, collection_name: str) -> None:
    if not hasattr(client, "get_collection"):
        raise ValueError("cannot validate existing collection schema")
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    sizes = {name: config.size for name, config in vectors.items()}
    if sizes != {"siglip": 1152, "jina": 1024}:
        raise ValueError("existing collection vector schema does not match")


def validate_collection(client, collection_name: str, expected_points: int = 177321) -> None:
    count = client.count(collection_name=collection_name, exact=True).count
    if count != expected_points:
        raise ValueError(f"collection has {count} points, expected {expected_points}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("keyframe_root", type=Path)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="AIC_data_aic25_b1")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    shards = discover_visual_shards(args.results_root)
    verify_and_extract_keyframes(shards, args.keyframe_root)
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    count = import_visual_shards(
        client,
        args.collection,
        shards,
        args.keyframe_root,
        resume_existing=args.resume_existing,
    )
    print(json.dumps({"collection": args.collection, "imported": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
