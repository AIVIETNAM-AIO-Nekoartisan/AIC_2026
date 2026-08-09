# AIC — Summary & How to Run

## What it is

A **multimodal video-retrieval system** for the HCMC AI Challenge (VBS-style interactive
search). Search a large Vietnamese news-video collection by **text/image**, **spoken words
(ASR)**, **on-screen text (OCR)**, and **temporal event sequences**, then submit
`(video, timestamp)` answers to the **DRES** judge under time pressure.

**Task types:** KIS (find one exact segment), Q&A, and TRAKE (find a video + align an
ordered sequence of events to keyframes).

## How it works (pipeline)

**Offline — build the indexes:**

```text
video ─> TransNetV2 shot detection ─> start/mid/end keyframes + GIFs
            ├─> SigLIP2 (1152d) + Jina-CLIP-v2 (1024d) ─> Qdrant  AIC_data
            ├─> caption + BGE-M3 (1024d)               ─> Qdrant  keyframe-captions
            └─> OCR text                                ─> Elasticsearch  ocr_data
audio ─> Whisper/VAD transcript                         ─> Elasticsearch  asr_data
```

**Online — live search:**

- **KIS:** optional GPT query expansion → SigLIP2/Jina embeddings → per-vector Qdrant
  search → **SRRF** fusion → optional **BLIP2** rerank.
- **Temporal / TRAKE:** GPT splits query into ordered events → per-event KIS+OCR+ASR fetch
  → normalize → **λ-decay beam search** (beam_size=8) → BLIP2 rerank.
- **Intelligent:** GPT decomposes the query into KIS/OCR/ASR legs → parallel search → weighted fusion.
- **Frontend:** Express proxy + vanilla-JS SPA — bookmarks (assemble TRAKE answers) and
  one-click DRES submit for all three answer formats.

**Stack:** Qdrant (vectors) · Elasticsearch (ASR/OCR) · FastAPI + CUDA models (backend) ·
Node/Express + vanilla JS (frontend).

## How to run

### Option A — Docker (all-in-one)

```bash
export HF_HOME=$(pwd)/temp_cache
docker compose up --build
```

Starts Qdrant `:6333`, Elasticsearch `:9200`, FastAPI `:8000`, web UI `:3000`.
GPU model loading needs the NVIDIA Container Toolkit.

### Option B — Manual (three processes, in order)

**1. Infrastructure (Qdrant + Elasticsearch):**

```bash
docker run -d --name aic_qdrant -p 6333:6333 \
  -v aic_qdrant_data:/qdrant/storage qdrant/qdrant

docker run -d --name aic_es -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.14.3
```

**2. Backend — run from the repo root** (running the file by path breaks `core.*` imports):

```bash
export HF_HOME=$(pwd)/temp_cache
python -m online.backend.api_server        # uvicorn on :8000, loads models (needs CUDA)
```

**3. Frontend:**

```bash
cd online/frontend && npm install && npm start   # http://localhost:3000
```

### Environment variables

- **Backend:** `OPENAI_API_KEY`, `QDRANT_URL`, `ES_HOST`, `ES_PORT`, `DISABLE_BLIP2`, `HF_HOME`
- **Frontend:** `API_URL`, `DRES_URL`, `DATA_ROOT`, `PORT`

### Tests (from repo root)

```bash
python3 -m unittest discover -s tests -v
```

Parsing tests run standalone; integration tests auto-skip when API/Qdrant/Elasticsearch are down.

> **Note:** the backend loads all ML models at startup and needs CUDA. Real search results
> require the offline pipeline to have already populated Qdrant + Elasticsearch.

---

*Full details (endpoint table, DRES answer formats, and the design debate on where to invest
for competition score) are in [`WORKFLOW.md`](WORKFLOW.md).*
