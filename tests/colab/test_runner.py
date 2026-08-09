import json
from pathlib import Path

import pytest
import cv2
import numpy as np
from zipfile import ZipFile

from aic_colab.config import load_config
from aic_colab.__main__ import main
from aic_colab.model_cache import create_model_cache, ensure_model_cache
from aic_colab.persistence import persist_verified
from aic_colab.runner import (
    RunContext,
    ShardJob,
    default_stage_handler,
    iter_dataset_jobs,
    prepare_model_cache,
    run_full,
    run_preflight,
)
from aic_colab.state import StateStore


def _context(tmp_path, jobs, handler, clock=lambda: 0.0):
    return RunContext(
        config=load_config(Path("colab/config/aic25-b1.json")),
        drive_root=tmp_path / "drive",
        work_root=tmp_path / "work",
        bundle_root=Path("colab"),
        state_store=StateStore(tmp_path / "state.json", "aic25-b1", "1"),
        jobs=lambda _ctx, _limit: iter(jobs),
        stage_handler=handler,
        clock=clock,
    )


def test_run_full_skips_verified_stage_and_finishes_remaining_stages(tmp_path):
    job = ShardJob("L21_a", "000000", (), ())
    source = tmp_path / "seed.bin"
    source.write_bytes(b"seed")
    artifact = persist_verified(source, tmp_path / "drive" / "seed.bin")
    calls = []

    def handler(ctx, current_job, stage):
        calls.append(stage)
        output = tmp_path / f"{stage}.bin"
        output.write_bytes(stage.encode())
        return [persist_verified(output, tmp_path / "drive" / f"{stage}.bin")]

    ctx = _context(tmp_path, [job], handler)
    state = ctx.state_store.load()
    ctx.state_store.mark(
        state,
        "L21_a",
        "extract",
        "000000",
        "done",
        artifacts=[str(artifact.path)],
    )
    ctx.state_store.save(state)

    report = run_full(ctx)

    assert report.status == "complete"
    assert calls == ["siglip", "jina", "validate"]


def test_run_full_pauses_before_starting_next_shard_at_deadline(tmp_path):
    jobs = [ShardJob("L21_a", "000000", (), ()), ShardJob("L21_a", "000001", (), ())]
    ticks = iter([0.0, 0.0, 31501.0])

    def handler(ctx, job, stage):
        output = tmp_path / f"{job.shard_id}-{stage}.bin"
        output.write_bytes(b"ok")
        return [persist_verified(output, tmp_path / "drive" / output.name)]

    ctx = _context(tmp_path, jobs, handler, clock=lambda: next(ticks, 31501.0))

    report = run_full(ctx, deadline_seconds=31500)

    assert report.status == "paused"
    assert report.completed_shards == 1


def test_model_cache_round_trip_and_rejects_corruption(tmp_path):
    source = tmp_path / "hf"
    (source / "models--example" / "snapshots" / "abc").mkdir(parents=True)
    (source / "models--example" / "snapshots" / "abc" / "config.json").write_text("{}")
    archive = tmp_path / "models.tar.zst"
    manifest_path = tmp_path / "manifest.json"
    created = create_model_cache(
        source,
        archive,
        manifest_path,
        models={"example/model": "abc"},
    )

    destination = tmp_path / "restored"
    ensure_model_cache(destination, archive, manifest_path)

    assert (destination / "models--example" / "snapshots" / "abc" / "config.json").is_file()
    assert json.loads(manifest_path.read_text())["sha256"] == created.sha256
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ensure_model_cache(tmp_path / "not-created", archive, manifest_path)


def test_default_pipeline_reads_archive_map_and_persists_tar_manifest(tmp_path):
    raw_video = tmp_path / "L21_V001.avi"
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64)
    )
    for value in range(3):
        writer.write(np.full((64, 64, 3), value * 30, dtype=np.uint8))
    writer.release()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    with ZipFile(drive_root / "Videos_L21_a.zip", "w") as archive:
        archive.write(raw_video, raw_video.name)
    bundle_root = tmp_path / "bundle"
    (bundle_root / "assets").mkdir(parents=True)
    with ZipFile(bundle_root / "assets" / "map-keyframes-aic25-b1.zip", "w") as archive:
        archive.writestr(
            "map-keyframes/L21_V001.csv",
            "n,pts_time,fps,frame_idx\n1,0,10,0\n2,0.2,10,2\n",
        )
    config = replace_config(
        load_config(Path("colab/config/aic25-b1.json")),
        archives=("Videos_L21_a.zip",),
        expected_videos=1,
        expected_keyframes=2,
        shard_size=2,
    )
    ctx = RunContext(
        config=config,
        drive_root=drive_root,
        work_root=tmp_path / "work",
        bundle_root=bundle_root,
        state_store=StateStore(tmp_path / "state.json", "aic25-b1", "1"),
        jobs=iter_dataset_jobs,
        stage_handler=default_stage_handler,
    )

    preflight = run_preflight(ctx)
    job = next(iter(iter_dataset_jobs(ctx, None)))
    artifacts = default_stage_handler(ctx, job, "extract")

    assert preflight["ready"]
    assert len(job.records) == 2
    assert len(artifacts) == 2
    assert all(artifact.path.is_file() for artifact in artifacts)


def replace_config(config, **changes):
    from dataclasses import replace

    return replace(config, **changes)


def test_status_cli_reports_new_state_without_data(tmp_path, capsys):
    code = main(["status", "--drive-root", str(tmp_path)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_id"] == "aic25-b1"
    assert payload["batches"] == {}


def test_prepare_model_cache_persists_once_and_restores_without_download(tmp_path):
    downloads = []

    def downloader(repo_id, cache_dir):
        downloads.append(repo_id)
        snapshot = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "rev1"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    ctx = _context(tmp_path, [], lambda *_args: [])
    ctx.model_downloader = downloader

    cache = prepare_model_cache(ctx)
    assert len(downloads) == 2
    assert (cache / "models--google--siglip2-so400m-patch14-384").is_dir()

    downloads.clear()
    import shutil

    shutil.rmtree(cache)
    restored = prepare_model_cache(ctx)
    assert downloads == []
    assert (restored / "models--jinaai--jina-clip-v2").is_dir()
