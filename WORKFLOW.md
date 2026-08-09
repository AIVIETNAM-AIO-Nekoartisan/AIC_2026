# AIC — Multimodal Video Retrieval: Workflow & Pipeline

## What this system is

This repository implements an interactive, VBS-style video-retrieval system for the HCMC AI Challenge, with results submitted to the DRES evaluation server under time pressure. It supports textual/visual Known-Item Search (KIS), Q&A, and TRAKE-style retrieval and alignment of ordered events. The collection is Vietnamese news video, so visual similarity, Vietnamese speech (ASR), and on-screen text (OCR) are all first-class signals.

## Architecture at a glance

| Component | Responsibility | Important files/services |
|---|---|---|
| `core/` | Shared model wrappers, database services, configuration, and parsing helpers | `core/models/`, `core/services/vector_store_optimized.py`, `core/services/elasticsearch_service.py` |
| `offline/` | Shot detection, keyframe/GIF extraction, embedding, caption, ASR, OCR, and index construction | `ingest_data.py`, `video_trans_detection.py`, `database_processing.py`, `ingest_captions.py`, `ingest_ASR_data.py`, `ingest_OCR_data.py` |
| `online/backend/` | FastAPI endpoints, model/resource lifecycle, KIS, temporal, intelligent, and agentic retrieval | `api_server.py`, `resources.py`, `routes.py`, `search_*.py`, `fusion.py`, `agent.py` |
| `online/frontend/` | Express proxy and vanilla-JS interactive search/submission UI | `server.js`, `public/index.html`, `public/app.js` |
| Infrastructure | Vector and lexical search | Qdrant on `:6333`; Elasticsearch on `:9200` |

At runtime, the browser talks to Express, Express proxies `/api/*` to FastAPI, and FastAPI queries Qdrant and Elasticsearch. Express also proxies `/dres/*` to DRES and serves indexed keyframes from `DATA_ROOT`.

## Offline indexing pipeline

```text
video
  └─> TransNetV2 shot detection
       └─> start / middle / end keyframes + GIF preview per shot
            ├─> SigLIP2 + Jina-CLIP-v2 ─> Qdrant AIC_data
            ├─> generated caption + BGE-M3 ─> Qdrant keyframe-captions
            └─> OCR text ─> Elasticsearch ocr_data

audio ─> Whisper/VAD transcript ─> Elasticsearch asr_data
```

1. `offline/ingest_data.py` orchestrates video ingestion. `offline/video_trans_detection.py` runs TransNetV2, thresholds transition probabilities, and saves three representative frames—start, middle, and end—for each sufficiently long shot. It also creates a short GIF preview.
2. `offline/database_processing.py` embeds every keyframe with:
   - SigLIP2 `google/siglip2-so400m-patch14-384` as named vector `siglip` (1152 dimensions).
   - Jina-CLIP-v2 `jinaai/jina-clip-v2` as named vector `jina` (1024 dimensions).
   - Optionally BEiT-3 as named vector `beit3` when its checkpoint is available.

   The vectors and frame/video/shot metadata are batch-upserted into Qdrant collection `AIC_data`. Deterministic point IDs are derived from video, shot number, and frame position.
3. `offline/ingest_captions.py` generates a caption with BLIP/BLIP2, embeds that text with BGE-M3, and writes a 1024-dimensional cosine vector to the separate `keyframe-captions` collection. It can caption one representative frame per shot and copy the caption to sibling keyframes. `offline/ingest_OCR_data.py` is another caption path: `core/models/MultiLLM_OCR.py` uses Gemini vision by default to produce OCR text and a retrieval-oriented caption; OCR text goes to Elasticsearch and captions can go through BGE-M3 to Qdrant.
4. `offline/ingest_ASR_data.py` and its parallel/enhanced variants use Whisper/VAD to extract Vietnamese transcripts and index them in Elasticsearch as `asr_data`.
5. `offline/ingest_OCR_data.py` indexes recognized on-screen text in Elasticsearch as `ocr_data`.

> **Caption implementation note:** the commit log references a Florence-2 caption index, but there is no Florence-2 implementation in the current codebase. The actual caption generators are BLIP/BLIP2 in `offline/ingest_captions.py` and Gemini in `core/models/MultiLLM_OCR.py`.

## Online retrieval pipeline

### KIS

`online/backend/search_kis.py` implements the main visual retrieval path:

```text
text and/or image query
  └─> optional GPT-4o translation/query expansion
       └─> SigLIP2 / Jina / optional BEiT-3 embeddings
            + BGE-M3 text embedding for caption search
             └─> per-vector Qdrant searches
                  └─> weighted SRRF fusion
                       └─> optional BLIP2 ITM/ITC top-k rerank
                            └─> keyframes returned to the UI
```

- SigLIP2 lowercases text, pads short queries to exactly 64 tokens, and window-averages longer queries. Jina provides multilingual and long-text coverage.
- SigLIP2, Jina, and optional BEiT-3 search named vectors in `AIC_data`; BGE-M3 searches `keyframe-captions`.
- `online/backend/fusion.py` min-max normalizes each result list and computes weighted Score-Reflected Reciprocal Rank Fusion:

  ```text
  SRRF(model, item) =
      weight(model) * ((1 - beta) / (k + rank) + beta * normalized_similarity)
  ```

  Scores are summed across models. `k` and model weights are request parameters; the call site currently hard-codes `beta = 0.6`, and default model weights are effectively equal.
- Optional BLIP2 reranking reorders the top candidate set using image-text matching (`itm`, more accurate) or image-text contrastive scoring (`itc`, faster).
- Qdrant payload filters can restrict a video or apply hard/soft object-count constraints.

### Temporal and TRAKE-oriented retrieval

`online/backend/search_temporal.py` asks GPT-4o to split a query into ordered events (or accepts explicit KIS/OCR/ASR fields). For each event it:

1. Retrieves KIS candidates plus OCR and ASR hits.
2. Enriches ASR hits with nearby Qdrant frames.
3. Min-max normalizes each modality and combines it with configurable KIS/OCR/ASR weights.
4. Groups candidates by video and runs chronological beam search with `beam_size = 8`.
5. Penalizes large event gaps with `lambda = max(exp(-decay_rate * gap_seconds), 0.001)`.
6. Optionally applies BLIP2 post-reranking and returns aligned event sequences.

The temporal algorithm can only sequence candidates present in each per-event pool; upstream keyframe coverage and retrieval recall therefore bound its result quality.

### Intelligent retrieval

`online/backend/search_intelligent.py` uses `GPT4oService.analyze_intelligent_query` to decompose a request into KIS, OCR, and ASR components and choose weights. It runs the enabled components in parallel; the KIS leg itself uses SRRF and optional BLIP2. Despite the module docstring calling the final step “RRF,” the current implementation actually min-max normalizes component scores and computes a weighted KIS/OCR/ASR sum, optionally requiring multi-source evidence from the same video.

### Agentic loop

`online/backend/agent.py` provides a stateless GPT-4o tool loop through `/agent/chat`. Over at most six tool rounds, it can call KIS, ASR, OCR, summarize result captions, and ask the operator a short clarifying question. The frontend retains and resends the conversation history.

### Key backend endpoints

| Endpoint | Purpose |
|---|---|
| `POST /search/kis` | Text/image KIS with encoder selection, SRRF weights, filtering, expansion, and reranking |
| `POST /search/audio` | ASR transcript search |
| `POST /search/ocr` | On-screen-text search |
| `POST /search/temporal/enhanced` | GPT-extracted temporal-event retrieval |
| `POST /search/temporal/combinate` | Temporal retrieval with explicit per-event KIS/OCR/ASR fields and weights |
| `POST /search/intelligent` | GPT decomposition followed by parallel multimodal retrieval |
| `POST /agent/chat` | Multi-round search-and-clarify agent |
| `POST /shot/detail` | Frames belonging to a shot |
| `GET /frame/info` | Frame payload plus nearby ASR/OCR context |
| `GET /health`, `GET /stats/{service}` | Service/model readiness and index statistics |

## Frontend & DRES submission

`online/frontend/server.js` is a thin Express server. It serves the vanilla-JS SPA, proxies backend requests, remaps stored keyframe paths under `DATA_ROOT`, supports video streaming, and forwards DRES calls to `{DRES_URL}/api/v2/*`.

The SPA provides KIS, ASR, OCR, Enhanced Temporal, Temporal Combinate, Intelligent, Combinate, and Browse tabs, plus a conversational agent mode. Result cards support pagination, GIF-hover previews, frame/video inspection, and `localStorage` bookmarks. Bookmarks assemble ordered frames for TRAKE; submitted-answer keys are also stored locally to reduce accidental duplicate submissions.

The UI emits the three DRES answer formats:

| Task | Payload |
|---|---|
| KIS | `{mediaItemName, start, end}` with `start`/`end` in milliseconds |
| Q&A | `{"text": "QA-<answer>-<video>-<milliseconds>"}` |
| TRAKE | `{"text": "TR-<video>-<frame1>,<frame2>,..."}` |

## How to run

### Docker Compose

```bash
export HF_HOME=$(pwd)/temp_cache
docker compose up --build
```

This starts Qdrant (`:6333`), Elasticsearch (`:9200`), FastAPI (`:8000`), and the web UI (`:3000`). GPU model loading requires a working NVIDIA Container Toolkit setup.

### Manual: three processes

1. Start infrastructure:

   ```bash
   docker run -d --name aic_qdrant --restart unless-stopped \
     -p 6333:6333 -v aic_qdrant_data:/qdrant/storage qdrant/qdrant

   docker run -d --name aic_es --restart unless-stopped \
     -p 9200:9200 \
     -e discovery.type=single-node \
     -e xpack.security.enabled=false \
     -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
     -v aic_es_data:/usr/share/elasticsearch/data \
     docker.elastic.co/elasticsearch/elasticsearch:8.14.3
   ```

2. From the repository root, start FastAPI:

   ```bash
   export HF_HOME=$(pwd)/temp_cache
   python -m online.backend.api_server
   ```

3. Start the frontend:

   ```bash
   cd online/frontend
   npm install
   npm start
   ```

Important environment variables:

- Backend/model: `OPENAI_API_KEY`, `QDRANT_URL`, `ES_HOST`, `ES_PORT`, `DISABLE_BLIP2`, `BLIP2_MODEL`, `BLIP2_BATCH`, `HF_HOME`.
- Frontend/proxy: `API_URL`, `DRES_URL`, `DATA_ROOT`, `VIDEO_DIRS`, `PORT`.

Run the test suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Standalone parsing tests run without services; integration tests skip when FastAPI, Qdrant, or Elasticsearch are unavailable.

## Design debate: where to invest for competition score

### Side A — strengthen the core retrieval engine first

- Every downstream workflow is recall-bounded by its candidate pool. Temporal search calls `process_kis_search` per event, and Intelligent search uses KIS as one of its three sources; better KIS retrieval therefore propagates to direct KIS, Intelligent, and TRAKE-oriented search.
- The ensemble is promising but not empirically tuned: SigLIP2 and Jina named vectors, BGE-M3 caption retrieval, SRRF, then BLIP2 reranking. `beta = 0.6` is hard-coded, weights default to parity, and the repository has no judged-query Recall@K/MRR/nDCG benchmark or tuning harness.
- Index construction fixes the recall ceiling. TransNetV2 retains only start/middle/end representatives, and `QdrantService.setup_collection()` accepts an existing collection without validating named-vector schema, per-point vector coverage, or stale/missing points.
- Recommended work: create a judged-query benchmark; ablate encoders and caption generators; grid-search SRRF and reranking parameters; audit and rebuild indexes with coverage checks.

### Side B — strengthen query understanding and the interactive/temporal workflow first

- DRES is interactive and penalizes wrong submissions. A relevant result is useful only if the operator can recognize, assemble, verify, and submit it quickly.
- The repository devotes substantial machinery to query interpretation and workflow: `GPT4o_service.py` expands queries, analyzes multimodal intent, and extracts up to six temporal events; `search_temporal.py` performs per-event fusion, chronological beam search, gap decay, and post-reranking; `agent.py` runs a multi-round search-and-clarify loop.
- Older commits also show investment in DeepSeek-based query recommendation/rewrite experiments, although that recommender is not present in the current source and current `HEAD` is a caption-fusion fix.
- The UI converts retrieval into competition points through GIF previews, pagination, bookmarks for TRAKE assembly, duplicate tracking, and one-click DRES submission for all three answer formats.
- The claim that encoder changes would yield only about 1–2% nDCG is a hypothesis, not a measured repository result; no benchmark currently exists to establish diminishing returns.

### Synthesis and recommended allocation

The two positions are complementary. Core recall sets the ceiling: no query decomposition, beam search, or UI can recover a shot that never enters a candidate pool. Workflow quality determines how much of that ceiling a human captures under time pressure: a good candidate still earns nothing if it cannot be found, verified, aligned, and submitted safely.

The first priority should be a small offline judged-query benchmark because it makes encoder, caption, fusion, temporal, and query-rewrite changes measurable. Once that foundation exists, use roughly a **40/60 retrieval-to-workflow split**: continue index audits and targeted retrieval ablations, while directing the larger share toward query decomposition, temporal/TRAKE alignment, result presentation, and submission safety—the most plausible sources of currently unrealized competition points.
