# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Multimodal video retrieval system for the **Ho Chi Minh City AI Challenge** ("Event Retrieval from Large Video Collection", a VBS-style interactive video-retrieval competition): search video keyframes by text/image (KIS), spoken words (ASR), on-screen text (OCR), and temporal event sequences, then submit answers to a DRES evaluation server. Code comments are largely in Vietnamese.

Competition context that shapes the design:
- Tasks follow the Video Browser Showdown format: **Textual/Visual KIS** (find one exact segment from a description or clip) and **Q&A**; answers are `(video, timestamp-range)` pairs submitted live under time pressure — hence the UI's bookmark flow, pagination-heavy result grids, and one-click DRES submission.
- **DRES** (Distributed Retrieval Evaluation Server) is the official judge; this project targets the AIC instance at `eventretrieval.oj.io.vn` (`/api/v2/login`, `/evaluation/list`, `/submit`). Wrong submissions incur penalties, so the UI tracks `submitted_answers` and shows scores/task status.
- The dataset is a large Vietnamese news-video collection (2024 edition: ~1,470 videos / ~330 hours), which is why ASR (Vietnamese speech) and OCR (on-screen Vietnamese text) are first-class search modalities alongside visual embeddings, and why temporal "event sequence" queries matter (news stories unfold across shots in one video).
- Task types as of the 2025 edition: Textual KIS, Video KIS, Q&A, and TRAKE (retrieve a video containing a described sequence of events and align each event to its keyframe — the bookmark flow exists to assemble TRAKE answers). 2026 keeps the LSC/VBS format, extends to multimedia lifelog data (images, audio, text) and adds an "automatic" (assistant-vs-assistant) track; prelims Aug 2026, finals Sep 12–26, 2026. The 2026 DRES instance/format was unannounced at time of writing — `DRES_URL` is env-configurable in `online/frontend/server.js` for when it changes.

## Running the system

Full stack via Docker: `docker compose up --build` → Qdrant :6333, Elasticsearch :9200, FastAPI backend :8000 (GPU via NVIDIA Container Toolkit), web UI :3000.

Manual, three processes in order:

1. **Infrastructure** — Qdrant (vectors) and Elasticsearch (ASR/OCR text) must be up:
   ```bash
   docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
   # Elasticsearch expected at localhost:9200
   ```
2. **Backend** — FastAPI server on port 8000 (loads all ML models at startup, needs CUDA). Env: `QDRANT_URL`, `ES_HOST`, `ES_PORT` (default localhost):
   ```bash
   python -m online.backend.api_server   # uvicorn app on 0.0.0.0:8000 (must run from repo root; running the file by path breaks `core.*` imports)
   ```
3. **Frontend** — Node.js web UI (`online/frontend/`: Express + vanilla JS, no build step). Env: `API_URL`, `DRES_URL`, `DATA_ROOT` (keyframe root for `/files`), `PORT`:
   ```bash
   cd online/frontend && npm install && npm start   # http://localhost:3000
   ```

`export HF_HOME=$(pwd)/temp_cache` before running to keep HuggingFace downloads local.

Tests (stdlib `unittest`, no pytest needed):
```bash
python3 -m unittest discover -s tests -v          # all
python3 -m unittest tests.test_parsing -v          # single module
```
Unit tests (`tests/test_parsing.py`) run standalone; integration tests (`tests/test_integration.py`) auto-skip when the API server / Qdrant / Elasticsearch are down. No linter configured.

## Architecture

Top-level layout: `core/` (shared code: model wrappers, DB services, helpers, config), `offline/` (indexing pipelines, run before the competition), `online/` (what runs live: `backend/` FastAPI + `frontend/` Node.js), `tests/`.

**online/frontend/** (frontend) — Node.js/Express (`server.js`) serving a vanilla-JS SPA (`public/`). The Express server is a thin proxy: `/api/*` → FastAPI backend, `/dres/*` → DRES server (browser can't call DRES directly due to CORS), `/files?p=` → keyframe images resolved under `DATA_ROOT` (remaps anything from `results/` onward, blocks path traversal). The SPA (light theme) mirrors the Streamlit feature set: KIS / ASR / OCR / Enhanced Temporal / Temporal Combinate / Intelligent / Combinate tabs, a service-status bar (Qdrant, ES, GPT, BLIP2, Jina on/off from `/health`), GIF-hover previews from `payload.gif_preview.path`, client-side pagination, localStorage bookmarks (also used to build TRAKE submissions), and DRES login/eval-picker/score-status/submit supporting all three answer formats: KIS `{mediaItemName,start,end}` (ms), QA `"QA-<ans>-<video>-<ms>"`, TRAKE `"TR-<video>-<f1>,<f2>,..."`. The DRES proxy is a generic passthrough: `/dres/<path>` → `{DRES_URL}/api/v2/<path>`.

**online/backend/** (backend) — FastAPI app split by concern: `api_server.py` (entrypoint: app + CORS + router), `schemas.py` (Pydantic models), `resources.py` (global `resources` object + `lifespan` init), `routes.py` (all endpoints, APIRouter), `search_kis.py` / `search_temporal.py` / `search_intelligent.py` (search pipelines), `fusion.py` (RRF/SRRF), `utils.py` (frame/timestamp helpers, ASR enrich, Qdrant lookups). All models and DB clients live in `resources`, populated in `lifespan`:
- SigLIP2 (`google/siglip2-so400m-patch14-384`) + Jina-CLIP-v2 (`jinaai/jina-clip-v2`, multilingual, 8192-token text) image/text embeddings → Qdrant collection `AIC_data` with **named vectors** (`siglip`=1152, `jina`=1024, optional `beit3`=1024); results fused with RRF; per-request `use_siglip`/`use_jina` toggles. SigLIP2 text quirks (transformers#43054): must lowercase + pad to exactly 64 tokens (both handled in `core/models/SigLIP_embedding.py`; queries >64 tokens are embedded via overlapping-window averaging). **transformers must stay <5** — jina-clip-v2's remote code breaks on 5.x
- BGE-M3 text embeddings → separate Qdrant collection `keyframe-captions` (1024, unnamed vector)
- Elasticsearch indexes `asr_data` and `ocr_data` via `core/services/elasticsearch_service.py`
- BLIP2 (`core/models/BLIP2_rerank.py`) reranks top-k results (itm/itc scoring)
- GPT-4o (`core/models/GPT4o_service.py`) for query expansion / intelligent search
- Every model init is wrapped in try/except — the server degrades gracefully when a model fails to load, so a missing weight file disables a feature rather than crashing

Key endpoints: `/search/kis`, `/search/audio`, `/search/ocr`, `/search/temporal/enhanced`, `/search/temporal/combinate`, `/search/intelligent`, `/shot/detail`, `/health`, `/stats/{service}`.

**core/services/** — `vector_store_optimized.py` (`QdrantService`, singleton via `get_qdrant_service`; the caption collection instantiates the class directly to bypass the singleton) and `elasticsearch_service.py`. Shared pure helpers (HMS timestamp parsing, frame-id extraction from payloads/filenames) live in `core/utils/parsing.py`; `core/config.py` loads `.env` (`OPENAI_API_KEY`, `QDRANT_URL`, etc.) but the backend mostly hardcodes its own collection/index names.

**core/models/** — one file per model wrapper. BEiT-3 needs checkpoints in `model_weights/` (`beit3_large_patch16_384_coco_retrieval.pth`, `beit3.spm`), which are gitignored.

**offline/** — indexing pipelines (argparse CLIs, run from repo root with `python -m offline.<name> --help`):
- `ingest_data.py` — videos → shot detection (TransNetV2 in `offline/video_trans_detection.py`) → keyframes/GIFs → embeddings (both `siglip` + `jina` vectors, via `offline/database_processing.py`) → Qdrant
- `ingest_ASR_data.py`, `ingest_OCR_data.py` and variants — audio transcripts / OCR text → Elasticsearch (and Qdrant for captions)
- Keyframe images land in `results/` and are referenced by payload paths
