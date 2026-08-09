# Colab Visual Encoding Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, package, and upload a self-contained Google Colab bundle that extracts 177,321 mapped keyframes from 873 videos, produces SigLIP2/Jina embedding shards with resumable Drive checkpoints, and imports the artifacts into a new local Qdrant collection with snapshot backup.

**Architecture:** Three thin notebooks invoke a tested `aic_colab` Python package. Every batch is copied from Drive to Colab local disk, extracted and persisted in 1,024-frame tar/NPZ shards with stable UUIDs and SHA-256 sidecars; Drive artifacts are the rebuildable source of truth. A separate local importer verifies those artifacts, extracts keyframes to Lexar, bulk-loads named vectors into `AIC_data_aic25_b1`, validates counts, and creates a Qdrant snapshot without modifying Elasticsearch.

**Tech Stack:** Python 3.11+, pytest, OpenCV, Pillow, NumPy, PyTorch, Transformers 4.x, Jina remote code, Google Colab notebooks, Google Drive mount, Qdrant client, Docker Compose.

## Global Constraints

- Dataset ID is exactly `aic25-b1`; full counts are 873 videos and 177,321 mapped keyframes.
- Use `google/siglip2-so400m-patch14-384` with 1,152-dimensional float32 normalized vectors.
- Use `jinaai/jina-clip-v2` with 1,024-dimensional float32 normalized vectors.
- Pin `transformers>=4.55,<5`; Jina remote code is incompatible with transformers 5.x.
- Use map `frame_idx` as extraction authority and `pts_time` as cross-check; do not run TransNetV2.
- Store WebP quality 90 and shard at a maximum of 1,024 keyframes.
- Stable point ID is `uuid.uuid5(uuid.NAMESPACE_URL, f"aic://{dataset_id}/{video_id}/{ordinal}")`.
- Persist local files before Drive copies; use `.partial`, `fsync`, SHA-256 verification, and atomic rename.
- Full notebook stops accepting work at 8h45m and exits by 9h with a valid checkpoint.
- Never delete input video ZIPs automatically.
- Bundle contains no `.env`, API keys, Drive tokens, Hugging Face tokens, GitHub PAT, or Qdrant credentials.
- Colab does not run Qdrant, Elasticsearch, OCR, ASR, caption, BGE, or object detection.
- Local import creates `AIC_data_aic25_b1`; it never overwrites or deletes an existing production collection.
- Work against design spec `docs/superpowers/specs/2026-08-09-colab-visual-encoding-bundle-design.md`.

---

### Task 1: Package configuration, inventory, map reader, and stable IDs

**Files:**
- Create: `colab/aic_colab/__init__.py`
- Create: `colab/aic_colab/config.py`
- Create: `colab/aic_colab/inventory.py`
- Create: `colab/aic_colab/map_reader.py`
- Create: `colab/aic_colab/ids.py`
- Create: `colab/config/aic25-b1.json`
- Create: `tests/colab/test_config_inventory.py`
- Create: `tests/colab/test_map_ids.py`

**Interfaces:**
- Produces `DatasetConfig`, `load_config(path: Path) -> DatasetConfig`, `EXPECTED_ARCHIVES`, `ArchiveInventory`, `inspect_archives(root: Path, expected: tuple[str, ...], test_zip: bool) -> ArchiveInventory`.
- Produces immutable `KeyframeRecord(video_id, ordinal, timestamp_seconds, fps, frame_index, point_id)`, `read_map_csv(path: Path, dataset_id: str) -> list[KeyframeRecord]`, and `stable_point_id(dataset_id: str, video_id: str, ordinal: int) -> str`.

- [ ] **Step 1: Write failing configuration and inventory tests**

```python
def test_config_has_all_archives_and_fixed_models(tmp_path):
    cfg = load_config(Path("colab/config/aic25-b1.json"))
    assert cfg.dataset_id == "aic25-b1"
    assert len(cfg.archives) == 14
    assert cfg.siglip_model == "google/siglip2-so400m-patch14-384"
    assert cfg.jina_model == "jinaai/jina-clip-v2"
    assert cfg.shard_size == 1024

def test_inventory_reports_missing_and_zero_byte_files(tmp_path):
    (tmp_path / "Videos_L21_a.zip").write_bytes(b"")
    report = inspect_archives(tmp_path, ("Videos_L21_a.zip", "Videos_L22_a.zip"), False)
    assert report.zero_byte == ("Videos_L21_a.zip",)
    assert report.missing == ("Videos_L22_a.zip",)
    assert not report.ready
```

- [ ] **Step 2: Run inventory tests and verify RED**

Run: `pytest -q tests/colab/test_config_inventory.py`

Expected: collection error because `aic_colab` does not exist.

- [ ] **Step 3: Implement config and inventory with immutable dataclasses**

Implement strict JSON parsing that rejects any dataset ID other than the file value, duplicate archive names, shard sizes outside 1–4096, and model dimensions other than 1152/1024. Inventory must sort output names and optionally run `ZipFile.testzip()`.

- [ ] **Step 4: Run inventory tests and verify GREEN**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_config_inventory.py`

Expected: all tests pass.

- [ ] **Step 5: Write failing map and ID tests**

```python
def test_stable_point_id_matches_design_example():
    assert stable_point_id("aic25-b1", "L21_V001", 1) == "9c6e7432-1908-54ea-8c96-8bbff30a1fc2"

def test_map_reader_converts_n_to_ordinal_and_rejects_non_monotonic_rows(tmp_path):
    path = tmp_path / "L21_V001.csv"
    path.write_text("n,pts_time,fps,frame_idx\n1,0.0,25,0\n2,1.0,25,25\n")
    rows = read_map_csv(path, "aic25-b1")
    assert [(r.ordinal, r.frame_index) for r in rows] == [(1, 0), (2, 25)]
    path.write_text("n,pts_time,fps,frame_idx\n1,1.0,25,25\n2,0.0,25,0\n")
    with pytest.raises(ValueError, match="monotonic"):
        read_map_csv(path, "aic25-b1")
```

- [ ] **Step 6: Run map tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_map_ids.py`

Expected: import failure for missing map/ID functions.

- [ ] **Step 7: Implement stable IDs and strict CSV mapping**

Require columns `n,pts_time,fps,frame_idx`, ordinal starting at one, strictly increasing ordinal/frame/timestamp, positive FPS, and non-negative frame index/timestamp.

- [ ] **Step 8: Run Task 1 tests and commit**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_config_inventory.py tests/colab/test_map_ids.py`

Expected: all tests pass.

Commit:

```bash
git add colab/aic_colab colab/config tests/colab/test_config_inventory.py tests/colab/test_map_ids.py
git commit -m "add Colab dataset inventory and mapping"
```

### Task 2: Atomic Drive persistence and recoverable state

**Files:**
- Create: `colab/aic_colab/persistence.py`
- Create: `colab/aic_colab/state.py`
- Create: `tests/colab/test_persistence_state.py`

**Interfaces:**
- Consumes `DatasetConfig` and stable batch/shard identifiers.
- Produces `sha256_file(path: Path) -> str`, `persist_verified(source: Path, destination: Path) -> Artifact`, `verify_artifact(path: Path) -> bool`, and `StateStore(path: Path, dataset_id: str, bundle_version: str)` with `load()`, `save(state)`, `mark(...)`, `reset_stale_running()`, and `rebuild_from_artifacts(results_root: Path)`.

- [ ] **Step 1: Write failing atomic persistence tests**

```python
def test_persist_verified_writes_final_file_and_sidecar(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    artifact = persist_verified(source, tmp_path / "drive" / "out.bin")
    assert artifact.path.read_bytes() == b"abc"
    assert artifact.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert artifact.path.with_suffix(".bin.sha256").read_text().strip() == artifact.sha256
    assert not list((tmp_path / "drive").glob("*.partial"))

def test_verify_artifact_rejects_tampering(tmp_path):
    source = tmp_path / "source.bin"; source.write_bytes(b"abc")
    artifact = persist_verified(source, tmp_path / "out.bin")
    artifact.path.write_bytes(b"changed")
    assert not verify_artifact(artifact.path)
```

- [ ] **Step 2: Run persistence tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_persistence_state.py -k persist`

Expected: import failure for `persistence`.

- [ ] **Step 3: Implement `.partial` copy, fsync, checksum sidecar, and atomic rename**

Use only same-directory temporary paths for rename atomicity. Never replace a valid destination unless its checksum differs and caller explicitly removes it first.

- [ ] **Step 4: Run persistence tests and verify GREEN**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_persistence_state.py -k persist`

Expected: persistence tests pass.

- [ ] **Step 5: Write failing state recovery tests**

```python
def test_state_store_resets_running_to_pending_and_restores_backup(tmp_path):
    store = StateStore(tmp_path / "state.json", "aic25-b1", "1")
    state = store.load()
    store.mark(state, "L21_a", "siglip", "000000", "running")
    store.save(state)
    (tmp_path / "state.json").write_text("broken")
    restored = store.load()
    assert restored["batches"]["L21_a"]["stages"]["siglip"]["shards"]["000000"]["status"] == "running"
    store.reset_stale_running(restored)
    assert restored["batches"]["L21_a"]["stages"]["siglip"]["shards"]["000000"]["status"] == "pending"
```

- [ ] **Step 6: Run state tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_persistence_state.py -k state`

Expected: missing `StateStore`.

- [ ] **Step 7: Implement versioned state, backup-before-save, and stale-running reset**

Reject mismatched dataset/bundle versions. State writes must preserve the previous valid JSON as `state.json.backup` and heartbeat writes must be independent from state writes.

- [ ] **Step 8: Run Task 2 tests and commit**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_persistence_state.py`

Expected: all tests pass.

Commit:

```bash
git add colab/aic_colab/persistence.py colab/aic_colab/state.py tests/colab/test_persistence_state.py
git commit -m "add verified Colab checkpoints"
```

### Task 3: Safe archive extraction, mapped frame extraction, and tar sharding

**Files:**
- Create: `colab/aic_colab/extraction.py`
- Create: `colab/aic_colab/sharding.py`
- Create: `tests/colab/test_extraction_sharding.py`

**Interfaces:**
- Consumes `KeyframeRecord` and archive paths.
- Produces `safe_extract_zip(archive: Path, destination: Path) -> list[Path]`, `extract_mapped_frames(video: Path, records: Sequence[KeyframeRecord], destination: Path, webp_quality: int = 90) -> list[ExtractedFrame]`, `partition_frames(frames, shard_size=1024)`, and `write_tar_shard(frames, tar_path: Path) -> TarShard`.

- [ ] **Step 1: Write failing ZIP-slip and mapped extraction tests**

Generate a 30-frame 320x240 MJPEG AVI fixture with OpenCV, write map rows for frame indexes 0, 10, and 29, and assert the three WebP files exist with ordinals 001–003. Create a malicious ZIP member `../escape.txt` and assert `safe_extract_zip` raises `ValueError("unsafe ZIP member")`.

- [ ] **Step 2: Run extraction tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_extraction_sharding.py -k 'extract or zip'`

Expected: missing extraction module.

- [ ] **Step 3: Implement sequential OpenCV decoding and safe ZIP extraction**

Decode from frame zero to the last mapped index once. Validate observed `CAP_PROP_POS_MSEC` against `max(1/fps, 0.05)` when OpenCV reports a positive timestamp; report exact video/ordinal/frame on failure. Write WebP through a temporary sibling then rename.

- [ ] **Step 4: Run extraction tests and verify GREEN**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_extraction_sharding.py -k 'extract or zip'`

Expected: extraction and ZIP tests pass.

- [ ] **Step 5: Write failing deterministic tar shard tests**

```python
def test_tar_shards_are_bounded_and_deterministic(extracted_frames, tmp_path):
    shards = partition_frames(extracted_frames, shard_size=2)
    assert [len(x) for x in shards] == [2, 1]
    first = write_tar_shard(shards[0], tmp_path / "000000.tar")
    with tarfile.open(first.path) as tf:
        assert tf.getnames() == ["L21/L21_V001/001.webp", "L21/L21_V001/002.webp"]
```

- [ ] **Step 6: Run tar tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_extraction_sharding.py -k tar`

Expected: missing sharding behavior.

- [ ] **Step 7: Implement deterministic tar metadata and 1,024-frame partitioning**

Set tar member uid/gid/mtime to zero and sort by `(video_id, ordinal)`. Member names must be relative and path-safe.

- [ ] **Step 8: Run Task 3 tests and commit**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_extraction_sharding.py`

Expected: all tests pass.

Commit:

```bash
git add colab/aic_colab/extraction.py colab/aic_colab/sharding.py tests/colab/test_extraction_sharding.py
git commit -m "add mapped keyframe extraction"
```

### Task 4: GPU encoder adapters, embedding shards, and validation

**Files:**
- Create: `colab/aic_colab/encoders.py`
- Create: `colab/aic_colab/validation.py`
- Create: `tests/colab/test_encoders_validation.py`

**Interfaces:**
- Produces protocol `ImageEncoder{name, dimension, encode(images) -> np.ndarray}`, concrete `SiglipEncoder`, `JinaEncoder`, `encode_with_oom_backoff(encoder, images, initial_batch_size) -> tuple[np.ndarray, int]`, `write_embedding_shard(point_ids, vectors, path)`, and `validate_embedding_shard(path, expected_ids, expected_dim) -> ValidationResult`.

- [ ] **Step 1: Write failing OOM backoff test with a real fake encoder**

```python
class ThresholdEncoder:
    name = "fake"; dimension = 4
    def encode(self, images):
        if len(images) > 2:
            raise torch.cuda.OutOfMemoryError("test")
        return np.ones((len(images), 4), dtype=np.float32)

def test_oom_halves_batch_and_returns_all_rows():
    vectors, used = encode_with_oom_backoff(ThresholdEncoder(), list(range(5)), 4)
    assert vectors.shape == (5, 4)
    assert used == 2
```

- [ ] **Step 2: Run encoder tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_encoders_validation.py -k oom`

Expected: missing encoder module.

- [ ] **Step 3: Implement lazy model adapters and bounded OOM backoff**

SigLIP uses `AutoProcessor`/`AutoModel.get_image_features`; Jina uses `AutoModel(..., trust_remote_code=True).encode_image`. Load only one model at a time, use fp16 CUDA inference, return normalized float32 CPU arrays, and retry at batch sizes `N, N/2, ... 1` at most three OOM reductions.

- [ ] **Step 4: Run OOM tests and verify GREEN**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_encoders_validation.py -k oom`

Expected: OOM test passes without loading a real model.

- [ ] **Step 5: Write failing NPZ schema/NaN/norm tests**

```python
def test_embedding_validation_checks_ids_shape_dtype_and_norm(tmp_path):
    ids = np.array(["a", "b"])
    vecs = np.array([[1,0,0,0], [0,1,0,0]], dtype=np.float32)
    path = write_embedding_shard(ids, vecs, tmp_path / "x.npz")
    assert validate_embedding_shard(path, ["a", "b"], 4).passed
    vecs[1, 0] = np.nan
    write_embedding_shard(ids, vecs, path)
    assert "NaN" in validate_embedding_shard(path, ["a", "b"], 4).errors[0]
```

- [ ] **Step 6: Run validation tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_encoders_validation.py -k validation`

Expected: missing validation functions.

- [ ] **Step 7: Implement compressed NPZ writing and strict validation**

Store only `point_ids` and `vectors`. Validate exact order, shape, float32 dtype, finite values, and row norms in `[0.999, 1.001]`.

- [ ] **Step 8: Run Task 4 tests and commit**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_encoders_validation.py`

Expected: all tests pass.

Commit:

```bash
git add colab/aic_colab/encoders.py colab/aic_colab/validation.py tests/colab/test_encoders_validation.py
git commit -m "add resumable visual encoders"
```

### Task 5: Orchestrator CLI, runtime deadline, reports, and model cache

**Files:**
- Create: `colab/aic_colab/runner.py`
- Create: `colab/aic_colab/model_cache.py`
- Create: `colab/aic_colab/__main__.py`
- Create: `tests/colab/test_runner.py`

**Interfaces:**
- Consumes all Task 1–4 interfaces.
- Produces `RunContext`, `run_preflight(ctx)`, `run_pilot(ctx, limit=1000)`, `run_full(ctx, deadline_seconds=31500)`, `ensure_model_cache(cache_root, drive_archive, manifest)`, and CLI commands `preflight`, `pilot`, `full`, `status`.

- [ ] **Step 1: Write failing runner resume/deadline tests**

Use a fake extractor and fake encoders injected into `RunContext`. Seed one verified tar/SigLIP artifact and assert `run_full` skips it, performs only Jina/validate, and marks the shard done. Use a fake monotonic clock crossing 31,500 seconds and assert the report status is `paused`, not `failed`.

- [ ] **Step 2: Run runner tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_runner.py -k 'resume or deadline'`

Expected: missing runner module.

- [ ] **Step 3: Implement dependency-injected orchestration and JSONL logging**

Runner stage order is `extract -> siglip -> jina -> validate`. Before skipping, verify every persisted checksum. Heartbeat interval is at most 60 seconds. Errors include batch/video/ordinal/stage/exception/attempts and do not expose environment values.

- [ ] **Step 4: Run resume/deadline tests and verify GREEN**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_runner.py -k 'resume or deadline'`

Expected: tests pass.

- [ ] **Step 5: Write failing model cache tests**

Create a fake Hugging Face cache directory and a small deterministic `.tar.zst` fixture with Python `tarfile` plus `zstandard`; verify extraction and checksum, corrupt one byte, and assert `ensure_model_cache` rejects it before extracting.

- [ ] **Step 6: Run model cache tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_runner.py -k model_cache`

Expected: missing cache function.

- [ ] **Step 7: Implement verified `tar.zst` model cache and CLI**

Use streaming Python `tarfile` plus `zstandard`, reject unsafe archive members before extraction, and avoid a dependency on the runtime's system `tar`/`zstd` binaries. Manifest pins both repo IDs and resolved revisions. Add exit codes: 0 complete/paused, 2 preflight data error, 3 artifact validation error, 4 model/runtime error.

- [ ] **Step 8: Run Task 5 tests and commit**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_runner.py`

Expected: all tests pass.

Commit:

```bash
git add colab/aic_colab/runner.py colab/aic_colab/model_cache.py colab/aic_colab/__main__.py tests/colab/test_runner.py
git commit -m "add Colab batch orchestrator"
```

### Task 6: Thin notebooks, Colab requirements, and operator guide

**Files:**
- Create: `colab/notebooks/00_Setup_Preflight.ipynb`
- Create: `colab/notebooks/01_Pilot_1000_Frames.ipynb`
- Create: `colab/notebooks/02_Encode_Full_Dataset.ipynb`
- Create: `colab/requirements-colab.txt`
- Create: `colab/README_COLAB.md`
- Create: `colab/VERSION`
- Create: `tests/colab/test_notebooks.py`

**Interfaces:**
- Notebooks call `python -m aic_colab preflight|pilot|full` and expose one editable `DRIVE_ROOT` cell.
- Guide documents exact upload/run/resume/download/database commands.

- [ ] **Step 1: Write failing notebook contract tests**

```python
@pytest.mark.parametrize("name,command", [
    ("00_Setup_Preflight.ipynb", "preflight"),
    ("01_Pilot_1000_Frames.ipynb", "pilot"),
    ("02_Encode_Full_Dataset.ipynb", "full"),
])
def test_notebook_is_valid_thin_runner(name, command):
    doc = json.loads((Path("colab/notebooks") / name).read_text())
    source = "\n".join("".join(c.get("source", [])) for c in doc["cells"])
    assert doc["nbformat"] == 4
    assert "drive.mount('/content/drive')" in source
    assert f"python -m aic_colab {command}" in source
    assert "DRIVE_ROOT" in source
    assert "GOOGLE_API_KEY" not in source and "HF_TOKEN" not in source
```

- [ ] **Step 2: Run notebook tests and verify RED**

Run: `pytest -q tests/colab/test_notebooks.py`

Expected: notebook files missing.

- [ ] **Step 3: Create minimal valid notebooks and pinned requirements**

Each notebook contains: purpose markdown, Drive mount, one `DRIVE_ROOT` config cell defaulting to `/content/drive/MyDrive/Video AIC`, bundle extraction/bootstrap, the one CLI invocation, and status/report display. Requirements include `transformers>=4.55,<5`, `accelerate==1.9.0`, `torch`, `opencv-python-headless`, `Pillow`, `numpy`, `sentencepiece`, `einops`, `timm==0.6.13`, `zstandard`, and `pytest`.

- [ ] **Step 4: Run notebook tests and verify GREEN**

Run: `pytest -q tests/colab/test_notebooks.py`

Expected: all three notebook contracts pass.

- [ ] **Step 5: Write README with exact operator commands**

Document preflight, pilot gate, full run, interpreting `safe_to_delete`, resuming after disconnect, recovering corrupt state, copying results with `rclone`, and keeping video ZIPs until both Drive and local validations pass.

- [ ] **Step 6: Run Markdown/file checks and commit**

Run: `python -m json.tool colab/notebooks/00_Setup_Preflight.ipynb >/dev/null && python -m json.tool colab/notebooks/01_Pilot_1000_Frames.ipynb >/dev/null && python -m json.tool colab/notebooks/02_Encode_Full_Dataset.ipynb >/dev/null && git diff --check`

Expected: exit 0.

Commit:

```bash
git add colab/notebooks colab/requirements-colab.txt colab/README_COLAB.md colab/VERSION tests/colab/test_notebooks.py
git commit -m "add Colab encoding notebooks"
```

### Task 7: Local Qdrant importer and snapshot backup

**Files:**
- Create: `offline/import_colab_visual.py`
- Create: `offline/backup_qdrant_collection.py`
- Create: `tests/test_colab_visual_import.py`

**Interfaces:**
- Consumes Drive result tree downloaded to Lexar.
- Produces `discover_visual_shards(results_root)`, `verify_and_extract_keyframes(...)`, `import_visual_shards(client, collection_name, shards, keyframe_root, batch_size=256)`, `validate_collection(client, collection_name, expected_points=177321)`, and `create_and_download_snapshot(qdrant_url, collection_name, destination)`.

- [ ] **Step 1: Write failing importer tests with a recording Qdrant client**

Create paired two-row SigLIP/Jina NPZ fixtures plus manifest/tar. Assert importer creates named vectors `siglip:1152`, `jina:1024`, uses the manifest point IDs, writes local payload file paths, upserts exactly two points, and refuses an existing collection unless `--resume-existing` targets the same schema.

- [ ] **Step 2: Run importer tests and verify RED**

Run: `pytest -q tests/test_colab_visual_import.py -k import`

Expected: missing importer module.

- [ ] **Step 3: Implement checksum-first keyframe extraction and Qdrant bulk load**

Reject path traversal in tar members. Create collection with named COSINE vectors and `indexing_threshold=0`; import 256 points per waited upsert; restore `indexing_threshold=20000` after all shards; create payload indexes for `frame.timestamp_seconds` and `video.name`.

- [ ] **Step 4: Run importer tests and verify GREEN**

Run: `pytest -q tests/test_colab_visual_import.py -k import`

Expected: importer tests pass.

- [ ] **Step 5: Write failing snapshot response/checksum tests**

Inject an HTTP transport returning a Qdrant snapshot name and bytes. Assert the file is saved under `<backup_root>/<collection>/<timestamp>/`, SHA-256 is recorded, and `backup-manifest.json` contains collection, points, Qdrant version, snapshot name, and bundle version.

- [ ] **Step 6: Run snapshot tests and verify RED**

Run: `pytest -q tests/test_colab_visual_import.py -k snapshot`

Expected: missing snapshot implementation.

- [ ] **Step 7: Implement REST snapshot create/download and backup manifest**

Use `urllib.request` with explicit timeouts and optional API-key header. Do not copy live `qdrant_data` or Docker `es_data` directories.

- [ ] **Step 8: Run Task 7 tests and commit**

Run: `pytest -q tests/test_colab_visual_import.py`

Expected: all tests pass.

Commit:

```bash
git add offline/import_colab_visual.py offline/backup_qdrant_collection.py tests/test_colab_visual_import.py
git commit -m "add Colab artifact database import"
```

### Task 8: Deterministic bundle build, full verification, and Drive upload

**Files:**
- Create: `colab/build_bundle.py`
- Create: `tests/colab/test_build_bundle.py`
- Modify: `.gitignore`
- Create at build time: `dist/aic25-colab-bundle-v1.zip`
- Create at build time: `dist/aic25-colab-bundle-v1.zip.sha256`

**Interfaces:**
- `build_bundle(repo_root: Path, map_archive: Path, output: Path) -> BundleArtifact` includes only approved Colab code/config/notebooks/tests/assets/docs plus the two local import/backup tools under `local_tools/`, fixed ZIP timestamps, sorted entries, and no secret/runtime/database files.

- [ ] **Step 1: Write failing deterministic/safety bundle tests**

```python
def test_bundle_is_deterministic_and_excludes_secrets(tmp_path):
    one = build_bundle(REPO, MAP_ARCHIVE, tmp_path / "one.zip")
    two = build_bundle(REPO, MAP_ARCHIVE, tmp_path / "two.zip")
    assert one.sha256 == two.sha256
    with ZipFile(one.path) as zf:
        names = set(zf.namelist())
    assert "assets/map-keyframes-aic25-b1.zip" in names
    assert "local_tools/import_colab_visual.py" in names
    assert "local_tools/backup_qdrant_collection.py" in names
    assert not any(".env" in n or "qdrant_data" in n or "es_data" in n for n in names)
```

- [ ] **Step 2: Run bundle tests and verify RED**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_build_bundle.py`

Expected: missing build module.

- [ ] **Step 3: Implement deterministic ZIP builder and `.gitignore` entries**

Reject map archives whose inventory is not exactly 873 CSV files and 177,321 data rows. Write output through a temporary file, create sidecar SHA-256, and add `/dist/`, `/qdrant_data/`, `/results/`, `/temp_cache/`, and `/colab/.runtime/` to `.gitignore` without removing existing rules.

- [ ] **Step 4: Run bundle tests and verify GREEN**

Run: `PYTHONPATH=colab pytest -q tests/colab/test_build_bundle.py`

Expected: bundle tests pass.

- [ ] **Step 5: Run the complete local verification suite**

Run:

```bash
PYTHONPATH=colab pytest -q
python -m compileall -q colab/aic_colab offline/import_colab_visual.py offline/backup_qdrant_collection.py
git diff --check
```

Expected: all tests pass, compileall and diff check exit 0.

- [ ] **Step 6: Build the production bundle from the real map archive**

Run:

```bash
python colab/build_bundle.py \
  --map-archive /media/nekoartisan/Lexar/AIC/videos/map-keyframes-aic25-b1.zip \
  --output dist/aic25-colab-bundle-v1.zip
sha256sum -c dist/aic25-colab-bundle-v1.zip.sha256
unzip -t dist/aic25-colab-bundle-v1.zip
```

Expected: map inventory reports 873 CSV/177,321 rows; checksum and ZIP tests pass.

- [ ] **Step 7: Commit source and tests**

```bash
git add colab/build_bundle.py tests/colab/test_build_bundle.py .gitignore
git commit -m "build deterministic Colab bundle"
```

- [ ] **Step 8: Upload and verify Drive metadata**

Create or reuse `Video AIC/AIC25-Colab`, upload the bundle ZIP and SHA-256 sidecar with the Google Drive connector, then list the folder and verify both filenames, exact byte sizes, private sharing state, and downloadable metadata. Do not upload `.env`, local database, model weights, video duplicates, or uncommitted artifacts.

- [ ] **Step 9: Final repository and Drive status report**

Run `git status --short --branch` and record bundle SHA-256, Drive folder URL, bundle URL, test counts, excluded modalities, four archive uploads still pending if applicable, and the exact first operator action (`00_Setup_Preflight.ipynb`).
