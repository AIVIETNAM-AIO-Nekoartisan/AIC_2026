from __future__ import annotations

import json
import csv
import io
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any

from .config import DatasetConfig
from .encoders import JinaEncoder, SiglipEncoder, encode_with_oom_backoff
from .extraction import ExtractedFrame, extract_mapped_frames, safe_extract_zip
from .inventory import inspect_archives
from .map_reader import KeyframeRecord
from .map_reader import read_map_csv
from .persistence import Artifact, persist_verified, verify_artifact
from .sharding import partition_frames, write_tar_shard
from .state import StateStore
from .validation import validate_embedding_shard, write_embedding_shard

import numpy as np
from PIL import Image
from zipfile import ZipFile


STAGES = ("extract", "siglip", "jina", "validate")


@dataclass(frozen=True)
class ShardJob:
    batch_id: str
    shard_id: str
    records: tuple[KeyframeRecord, ...]
    frames: tuple[Any, ...]


@dataclass
class RunContext:
    config: DatasetConfig
    drive_root: Path
    work_root: Path
    bundle_root: Path
    state_store: StateStore
    jobs: Callable[["RunContext", int | None], Iterable[ShardJob]]
    stage_handler: Callable[["RunContext", ShardJob, str], list[Artifact]]
    clock: Callable[[], float] = time.monotonic
    namespace: str = "results"
    siglip_factory: Callable[[], Any] | None = None
    jina_factory: Callable[[], Any] | None = None


@dataclass(frozen=True)
class RunReport:
    status: str
    completed_shards: int
    skipped_stages: int
    elapsed_seconds: float


def run_preflight(ctx: RunContext) -> dict[str, Any]:
    inventory = inspect_archives(ctx.drive_root, ctx.config.archives, test_zip=True)
    map_archive = ctx.bundle_root / "assets" / "map-keyframes-aic25-b1.zip"
    videos, keyframes = _map_inventory(map_archive)
    map_ready = (
        videos == ctx.config.expected_videos
        and keyframes == ctx.config.expected_keyframes
    )
    report = {
        "ready": inventory.ready and map_ready,
        "archives": {
            "present": list(inventory.present),
            "missing": list(inventory.missing),
            "zero_byte": list(inventory.zero_byte),
            "corrupt": list(inventory.corrupt),
        },
        "map": {"videos": videos, "keyframes": keyframes, "ready": map_ready},
    }
    path = ctx.drive_root / "AIC25-Colab" / "runtime" / "preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _map_inventory(path: Path) -> tuple[int, int]:
    videos = 0
    rows = 0
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            videos += 1
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                rows += sum(1 for _ in reader)
    return videos, rows


def _extract_once(archive: Path, destination: Path) -> Path:
    marker = destination / ".complete"
    if marker.is_file():
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    safe_extract_zip(archive, destination)
    marker.write_text("ok\n")
    return destination


def iter_dataset_jobs(
    ctx: RunContext, limit: int | None
) -> Iterable[ShardJob]:
    maps_root = _extract_once(
        ctx.bundle_root / "assets" / "map-keyframes-aic25-b1.zip",
        ctx.work_root / "maps",
    )
    maps = {path.stem: path for path in maps_root.rglob("*.csv")}
    remaining = limit
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    for archive_name in ctx.config.archives:
        if remaining is not None and remaining <= 0:
            break
        batch_id = Path(archive_name).stem.removeprefix("Videos_")
        video_root = _extract_once(
            ctx.drive_root / archive_name,
            ctx.work_root / "archives" / batch_id,
        )
        pending: list[ExtractedFrame] = []
        shard_number = 0
        for video in sorted(
            (path for path in video_root.rglob("*") if path.suffix.lower() in video_extensions),
            key=lambda item: item.name,
        ):
            map_path = maps.get(video.stem)
            if map_path is None:
                raise ValueError(f"missing map CSV for {video.stem}")
            records = read_map_csv(map_path, ctx.config.dataset_id)
            if remaining is not None:
                records = records[:remaining]
            frames = extract_mapped_frames(
                video,
                records,
                ctx.work_root / "frames" / batch_id,
                ctx.config.webp_quality,
            )
            pending.extend(frames)
            if remaining is not None:
                remaining -= len(frames)
            while len(pending) >= ctx.config.shard_size:
                current = pending[: ctx.config.shard_size]
                pending = pending[ctx.config.shard_size :]
                yield _job(batch_id, shard_number, current)
                shard_number += 1
            if remaining is not None and remaining <= 0:
                break
        if pending:
            yield _job(batch_id, shard_number, pending)


def _job(batch_id: str, shard_number: int, frames: list[ExtractedFrame]) -> ShardJob:
    ordered = sorted(frames, key=lambda item: (item.record.video_id, item.record.ordinal))
    return ShardJob(
        batch_id,
        f"{shard_number:06d}",
        tuple(item.record for item in ordered),
        tuple(ordered),
    )


def _results_root(ctx: RunContext) -> Path:
    return ctx.drive_root / "AIC25-Colab" / ctx.namespace / ctx.config.dataset_id


def default_stage_handler(
    ctx: RunContext, job: ShardJob, stage: str
) -> list[Artifact]:
    results = _results_root(ctx)
    local = ctx.work_root / "artifacts" / job.batch_id
    local.mkdir(parents=True, exist_ok=True)
    if stage == "extract":
        tar_local = local / f"{job.shard_id}.tar"
        if tar_local.exists():
            tar_local.unlink()
        write_tar_shard(job.frames, tar_local)
        manifest_local = local / f"{job.shard_id}.jsonl"
        with manifest_local.open("w", encoding="utf-8") as handle:
            for frame in job.frames:
                record = frame.record
                handle.write(
                    json.dumps(
                        {
                            "dataset_id": ctx.config.dataset_id,
                            "batch_id": job.batch_id,
                            "video_id": record.video_id,
                            "ordinal": record.ordinal,
                            "timestamp_seconds": record.timestamp_seconds,
                            "fps": record.fps,
                            "frame_index": record.frame_index,
                            "point_id": record.point_id,
                            "keyframe_shard": f"keyframes/{job.batch_id}/{job.shard_id}.tar",
                            "keyframe_member": frame.member_name,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        return [
            persist_verified(
                tar_local, results / "keyframes" / job.batch_id / tar_local.name
            ),
            persist_verified(
                manifest_local,
                results / "manifests" / job.batch_id / manifest_local.name,
            ),
        ]
    if stage in {"siglip", "jina"}:
        images = _load_tar_images(
            results / "keyframes" / job.batch_id / f"{job.shard_id}.tar"
        )
        if stage == "siglip":
            encoder = (ctx.siglip_factory or (lambda: SiglipEncoder(
                ctx.config.siglip_model, ctx.config.siglip_dimension
            )))()
            batch_size = 32
        else:
            encoder = (ctx.jina_factory or (lambda: JinaEncoder(
                ctx.config.jina_model, ctx.config.jina_dimension
            )))()
            batch_size = 16
        try:
            vectors, _ = encode_with_oom_backoff(encoder, images, batch_size)
        finally:
            close = getattr(encoder, "close", None)
            if close:
                close()
            for image in images:
                image.close()
        output = local / f"{job.shard_id}.{stage}.npz"
        write_embedding_shard([record.point_id for record in job.records], vectors, output)
        return [persist_verified(output, results / "embeddings" / job.batch_id / output.name)]
    if stage == "validate":
        errors: list[str] = []
        ids = [record.point_id for record in job.records]
        for model, dimension in (
            ("siglip", ctx.config.siglip_dimension),
            ("jina", ctx.config.jina_dimension),
        ):
            path = results / "embeddings" / job.batch_id / f"{job.shard_id}.{model}.npz"
            result = validate_embedding_shard(path, ids, dimension)
            errors.extend(result.errors)
        if errors:
            raise ValueError("; ".join(errors))
        report_local = local / f"{job.shard_id}.validation.json"
        report_local.write_text(
            json.dumps({"status": "passed", "count": len(ids)}, sort_keys=True) + "\n"
        )
        return [
            persist_verified(
                report_local,
                results / "reports" / job.batch_id / report_local.name,
            )
        ]
    raise ValueError(f"unknown stage: {stage}")


def _load_tar_images(path: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read {member.name}")
            with source:
                image = Image.open(io.BytesIO(source.read())).convert("RGB")
                image.load()
                images.append(image)
    return images


def _stage_state(state: dict[str, Any], job: ShardJob, stage: str) -> dict[str, Any]:
    return (
        state.get("batches", {})
        .get(job.batch_id, {})
        .get("stages", {})
        .get(stage, {})
        .get("shards", {})
        .get(job.shard_id, {})
    )


def _stage_is_reusable(state: dict[str, Any], job: ShardJob, stage: str) -> bool:
    record = _stage_state(state, job, stage)
    artifacts = record.get("artifacts", [])
    return record.get("status") == "done" and bool(artifacts) and all(
        verify_artifact(Path(path)) for path in artifacts
    )


def run_full(ctx: RunContext, deadline_seconds: int = 31500) -> RunReport:
    started = ctx.clock()
    state = ctx.state_store.load()
    ctx.state_store.reset_stale_running(state)
    ctx.state_store.save(state)
    completed = 0
    skipped = 0
    paused = False
    for job in ctx.jobs(ctx, None):
        if ctx.clock() - started >= deadline_seconds:
            paused = True
            break
        for stage in STAGES:
            if _stage_is_reusable(state, job, stage):
                skipped += 1
                continue
            ctx.state_store.mark(
                state, job.batch_id, stage, job.shard_id, "running"
            )
            ctx.state_store.save(state)
            try:
                artifacts = ctx.stage_handler(ctx, job, stage)
            except Exception as exc:
                ctx.state_store.mark(
                    state,
                    job.batch_id,
                    stage,
                    job.shard_id,
                    "error",
                    error={"class": type(exc).__name__, "message": str(exc)},
                )
                ctx.state_store.save(state)
                _append_event(ctx, "errors.jsonl", job, stage, exc)
                raise
            ctx.state_store.mark(
                state,
                job.batch_id,
                stage,
                job.shard_id,
                "done",
                artifacts=[str(artifact.path) for artifact in artifacts],
                checksums=[artifact.sha256 for artifact in artifacts],
            )
            ctx.state_store.save(state)
            _append_event(ctx, "events.jsonl", job, stage, None)
        completed += 1
    elapsed = ctx.clock() - started
    report = RunReport("paused" if paused else "complete", completed, skipped, elapsed)
    _write_report(ctx, report)
    return report


def run_pilot(ctx: RunContext, limit: int = 1000) -> RunReport:
    original = ctx.jobs
    original_namespace = ctx.namespace
    original_store = ctx.state_store
    ctx.jobs = lambda current, _limit: original(current, limit)
    ctx.namespace = "pilot"
    ctx.state_store = StateStore(
        ctx.drive_root / "AIC25-Colab" / "runtime" / "pilot-state.json",
        ctx.config.dataset_id,
        original_store.bundle_version,
    )
    try:
        report = run_full(ctx, deadline_seconds=ctx.config.deadline_seconds)
        pilot_report = ctx.drive_root / "AIC25-Colab" / "pilot" / "report.json"
        pilot_report.parent.mkdir(parents=True, exist_ok=True)
        pilot_report.write_text(
            json.dumps(
                {
                    **report.__dict__,
                    "status": "passed" if report.status == "complete" else report.status,
                    "bundle_version": original_store.bundle_version,
                    "limit": limit,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return report
    finally:
        ctx.jobs = original
        ctx.namespace = original_namespace
        ctx.state_store = original_store


def _append_event(
    ctx: RunContext, filename: str, job: ShardJob, stage: str, error: Exception | None
) -> None:
    path = ctx.drive_root / "AIC25-Colab" / "runtime" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "batch": job.batch_id,
        "shard": job.shard_id,
        "stage": stage,
        "status": "error" if error else "done",
    }
    if error:
        payload["exception"] = type(error).__name__
        payload["message"] = str(error)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_report(ctx: RunContext, report: RunReport) -> None:
    path = ctx.drive_root / "AIC25-Colab" / "runtime" / "last-run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.__dict__, indent=2, sort_keys=True) + "\n")
