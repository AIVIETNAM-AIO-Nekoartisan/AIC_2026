/* VBS web UI — vanilla JS, không framework. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const PAGE_SIZE = 24;
const state = {
  dres: JSON.parse(localStorage.getItem("dres") || "null"), // {sessionId, username, evaluationId}
  submitted: JSON.parse(localStorage.getItem("submitted") || "[]"), // [{key, time, ok}]
  bookmarks: JSON.parse(localStorage.getItem("bookmarks") || "[]"), // [{video, frame, ts}]
  results: [],
  page: 0,
};

// ============ helpers ============
async function api(path, body) {
  const r = await fetch(`/api${path}`, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status} ${await r.text()}`);
  return r.json();
}

async function dres(path, body) {
  const r = await fetch(`/dres/${path}`, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  return { ok: r.ok, status: r.status, data };
}

function fmtTime(s) {
  if (s == null || isNaN(s)) return "";
  const m = Math.floor(s / 60), sec = (s % 60).toFixed(1);
  return `${m}:${String(sec).padStart(4, "0")}`;
}

// Duck-type mọi kiểu kết quả (KIS/ASR/OCR/temporal-step) về cùng một meta.
function itemMeta(item) {
  const p = item.payload || item;
  const video = p.video?.name || item.video_name || item.video || null;
  const frame = p.frame?.index ?? item.frame_id ?? item.frame ?? null;
  let ts = p.frame?.timestamp_seconds ?? item.timestamp_seconds ?? item.timestamp ?? item.ts ?? null;
  if (ts != null) ts = parseFloat(ts);
  const filePath = item.file_path || p.file_path || null;
  const gifPath = p.gif_preview?.path || null;
  const text = item.text || p.text || item.ocr_text || item.asr_text || null;
  const caption = p.caption || item.caption || null;
  const score = item.score ?? item.rerank_score ?? item.enhanced_score ?? null;
  return { video, frame, ts, filePath, gifPath, text, caption, score };
}

function setStatus(msg) { $("#status").textContent = msg || ""; $("#toolbar").classList.remove("hidden"); }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

// Enter = tìm ngay (Shift+Enter mới xuống dòng trong ô mô tả nhiều dòng)
document.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
  const el = e.target;
  if (!(el instanceof HTMLTextAreaElement) && !(el instanceof HTMLInputElement)) return;
  if (el.type === "file" || el.type === "checkbox") return;
  const form = el.form || document.querySelector(".search-form:not(.hidden)");
  if (!form) return;
  e.preventDefault();
  form.requestSubmit();
});

// ============ service status (bật/tắt các thành phần) ============
const SERVICE_LABELS = {
  qdrant: "Qdrant", es_audio: "ES·ASR", es_ocr: "ES·OCR",
  siglip: "SigLIP", jina: "Jina", bge: "BGE",
  gpt: "GPT-4o", reranker: "BLIP2", beit3: "BEiT-3",
};
const SERVICE_MODEL_KEYS = { reranker: "blip2" };
async function refreshServices() {
  const el = $("#services");
  try {
    const h = await api("/health");
    const svcs = h.services || {};
    const models = h.models || {};
    const errors = h.errors || {};
    const apiReady = !!h.kis_ready;
    const apiLabel = apiReady ? "API ✔" : `API · ${h.phase || "warming-up"}`;
    el.innerHTML = `<span class="pill ${apiReady ? "pill-on" : "pill-off"}">${apiLabel}</span>` +
      Object.entries(SERVICE_LABELS).map(([k, label]) => {
        if (!(k in svcs)) return "";
        const on = !!svcs[k];
        const modelKey = SERVICE_MODEL_KEYS[k] || k;
        const modelState = models[modelKey];
        const loading = modelState === "pending" || modelState === "loading";
        const suffix = on ? "✔" : loading ? "…" : "✕";
        const detail = errors[modelKey] || (loading ? "đang tải" : on ? "sẵn sàng" : modelState || "không khả dụng");
        return `<span class="pill ${on ? "pill-on" : "pill-off"}" title="${label}: ${escapeHtml(detail)}">${label} ${suffix}</span>`;
      }).join("");
  } catch {
    el.innerHTML = `<span class="pill pill-off">API ✕ (backend không phản hồi)</span>`;
  }
}
refreshServices();
setInterval(refreshServices, 15000);

// ============ tabs ============
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".search-form, .opt-group").forEach((f) =>
      f.classList.toggle("hidden", f.dataset.tab !== btn.dataset.tab));
  });
});

// ============ search: KIS ============
async function fileToBase64(file) {
  const buf = await file.arrayBuffer();
  let bin = "";
  new Uint8Array(buf).forEach((b) => (bin += String.fromCharCode(b)));
  return btoa(bin);
}

$("#form-kis").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    text_query: f.text_query.value.trim() || null,
    num_results: +f.num_results.value,
    score_threshold: +f.score_threshold.value,
    use_rerank: f.use_rerank.checked,
    rerank_score_type: f.rerank_score_type.value,
    rerank_top_k: +f.rerank_top_k.value,
    auto_expand: f.auto_expand.checked,
    num_expansions: +f.num_expansions.value,
    use_siglip: f.use_siglip.checked,
    use_jina: f.use_jina.checked,
    use_beit3: f.use_beit3.checked,
    use_bge: f.use_bge.checked,
    rrf_weights: {
      siglip: +f.w_siglip.value,
      jina: +f.w_jina.value,
      beit3: +f.w_beit3.value,
      caption: +f.w_caption.value,
    },
  };
  // Lọc theo object detection (nếu người dùng nhập)
  const objSpecs = parseObjectSpecs(f.object_class.value);
  if (objSpecs.length) body.object_filters = objSpecs;
  body.object_count_tolerance = +f.object_tolerance.value;
  body.object_filter_mode = f.object_mode.value;
  body.object_score_floor = +f.object_floor.value;
  if (f.image.files[0]) body.image_base64 = await fileToBase64(f.image.files[0]);
  if (!body.text_query && !body.image_base64) return setStatus("Nhập mô tả hoặc chọn ảnh.");
  await runSearch("/search/kis", body);
});
$("#form-kis").image.addEventListener("change", (e) => {
  $("#image-name").textContent = e.target.files[0] ? `✓ ${e.target.files[0].name}` : "";
});

// ============ search: ASR / OCR ============
$("#form-asr").addEventListener("submit", (e) => {
  e.preventDefault();
  const f = e.target;
  if (!f.query.value.trim()) return setStatus("Nhập nội dung ASR.");
  runSearch("/search/audio", { query: f.query.value.trim(), num_results: +f.num_results.value });
});

$("#form-ocr").addEventListener("submit", (e) => {
  e.preventDefault();
  const f = e.target;
  if (!f.query.value.trim()) return setStatus("Nhập nội dung OCR.");
  runSearch("/search/ocr", { query: f.query.value.trim(), num_results: +f.num_results.value });
});

// ============ search: Enhanced Temporal ============
$("#form-temporal").addEventListener("submit", (e) => {
  e.preventDefault();
  const f = e.target;
  if (!f.query.value.trim()) return setStatus("Nhập mô tả chuỗi sự kiện.");
  runSearch("/search/temporal/enhanced", {
    query: f.query.value.trim(),
    num_results: +f.num_results.value,
    per_event_k: +f.per_event_k.value,
    max_events: +f.max_events.value,
    decay_rate: +f.decay_rate.value,
    same_video_only: f.same_video_only.checked,
    max_gap_seconds: f.max_gap_seconds.value ? +f.max_gap_seconds.value : null,
  });
});

// ============ search: Temporal Combinate ============
function addEventRow(vals = {}) {
  const n = $("#events").children.length + 1;
  const div = document.createElement("div");
  div.className = "event-row";
  div.innerHTML = `
    <span class="ev-num">${n}</span>
    <input placeholder="KIS" class="ev-kis" value="${vals.kis || ""}">
    <input placeholder="OCR" class="ev-ocr" value="${vals.ocr || ""}">
    <input placeholder="ASR" class="ev-asr" value="${vals.asr || ""}">
    <button type="button" class="ev-del" title="Xóa event">×</button>`;
  div.querySelector(".ev-del").addEventListener("click", () => {
    div.remove();
    [...$("#events").children].forEach((r, i) => (r.querySelector(".ev-num").textContent = i + 1));
  });
  $("#events").appendChild(div);
}
$("#add-event").addEventListener("click", () => addEventRow());
addEventRow(); addEventRow();

$("#form-tcomb").addEventListener("submit", (e) => {
  e.preventDefault();
  const f = e.target;
  const events = [...document.querySelectorAll(".event-row")].map((r) => ({
    kis: r.querySelector(".ev-kis").value.trim() || null,
    ocr: r.querySelector(".ev-ocr").value.trim() || null,
    asr: r.querySelector(".ev-asr").value.trim() || null,
  })).filter((ev) => ev.kis || ev.ocr || ev.asr);
  if (!events.length) return setStatus("Thêm ít nhất 1 event có nội dung.");
  runSearch("/search/temporal/combinate", {
    query: events.map((ev) => ev.kis || ev.ocr || ev.asr).join(" ; "),
    events_explicit: events,
    num_results: +f.num_results.value,
    per_event_k: +f.per_event_k.value,
    weights: { kis: +f.w_kis.value, ocr: +f.w_ocr.value, asr: +f.w_asr.value },
    use_rerank_per_event: f.use_rerank_per_event.checked,
  });
});

// ============ search: Intelligent ============
$("#form-intelligent").addEventListener("submit", (e) => {
  e.preventDefault();
  const f = e.target;
  if (!f.query.value.trim()) return setStatus("Nhập query.");
  runSearch("/search/intelligent", {
    query: f.query.value.trim(),
    num_results: +f.num_results.value,
    score_threshold: +f.score_threshold.value,
    enable_kis: f.enable_kis.checked,
    enable_ocr: f.enable_ocr.checked,
    enable_asr: f.enable_asr.checked,
    show_analysis: true,
  });
});

// ============ search: Combinate (fuse client-side, như Streamlit) ============
$("#form-comb").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const kisQ = f.kis_q.value.trim(), ocrQ = f.ocr_q.value.trim(), asrQ = f.asr_q.value.trim();
  if (!kisQ && !ocrQ && !asrQ) return setStatus("Nhập ít nhất 1 query (KIS/OCR/ASR).");
  const n = +f.num_results.value;
  setStatus("Đang truy vấn & gộp kết quả...");
  try {
    const [kis, ocr, asr] = await Promise.all([
      kisQ ? api("/search/kis", { text_query: kisQ, num_results: n * 3, score_threshold: 0, use_rerank: false, auto_expand: false }) : { results: [] },
      ocrQ ? api("/search/ocr", { query: ocrQ, num_results: Math.min(n * 2, 100) }) : { results: [] },
      asrQ ? api("/search/audio", { query: asrQ, num_results: Math.min(n * 2, 100) }) : { results: [] },
    ]);
    // RRF fuse theo file_path
    const K = 60, fused = new Map();
    for (const [list, w] of [[kis.results, 1.0], [ocr.results, 0.8], [asr.results, 0.8]]) {
      (list || []).forEach((item, rank) => {
        const key = item.file_path || JSON.stringify(itemMeta(item));
        const cur = fused.get(key) || { item, rrf: 0, hits: 0 };
        cur.rrf += w / (K + rank + 1);
        cur.hits += 1;
        fused.set(key, cur);
      });
    }
    const merged = [...fused.values()]
      .sort((a, b) => b.hits - a.hits || b.rrf - a.rrf)
      .slice(0, n)
      .map((x) => ({ ...x.item, score: x.rrf }));
    state.results = merged; state.page = 0;
    setStatus(`${merged.length} kết quả (gộp ${kis.results?.length || 0} KIS + ${ocr.results?.length || 0} OCR + ${asr.results?.length || 0} ASR)`);
    renderPage();
  } catch (err) { setStatus(`Lỗi: ${err.message}`); }
});

// ============ browse: theo video / shot / thời gian ============
$("#form-browse").mode.addEventListener("change", (e) => {
  document.querySelector(".browse-time").classList.toggle("hidden", e.target.value !== "time");
  document.querySelector(".browse-shot").classList.toggle("hidden", e.target.value !== "shot");
});

$("#form-browse").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const video = f.video.value.trim();
  if (!video) return setStatus("Nhập tên video (vd: L27_V001).");
  setStatus("Đang tải frames...");
  showSkeletons();
  try {
    if (f.mode.value === "time") {
      const qs = `start_s=${+f.start_s.value}&end_s=${+f.end_s.value}&limit=${+f.limit.value}&sample_stride=${+f.sample_stride.value}`;
      const res = await api(`/videos/${encodeURIComponent(video)}/frames?${qs}`);
      state.results = res.frames || [];
      setStatus(`${state.results.length} frame — ${video}, ${f.start_s.value}s → ${f.end_s.value}s`);
    } else {
      const qs = `start_shot=${+f.start_shot.value}&end_shot=${+f.end_shot.value}`;
      const res = await api(`/videos/${encodeURIComponent(video)}/shots?${qs}`);
      // mỗi shot -> 1 hàng sequence gồm 3 frame start/middle/end
      state.results = (res.shots || []).map((s) => ({
        label: `Shot ${s.shot_number}`,
        video: s.video_name,
        steps: ["start", "middle", "end"].map((k) => s.positions?.[k]).filter(Boolean),
      }));
      setStatus(`${state.results.length} shot — ${video}, shot ${f.start_shot.value} → ${f.end_shot.value}`);
    }
    state.page = 0;
    renderPage();
  } catch (err) {
    $("#results").innerHTML = "";
    setStatus(`Lỗi: ${err.message}`);
  }
});

// ============ run & render ============
function showSkeletons(n = 12) {
  const grid = document.createElement("div");
  grid.className = "grid";
  for (let i = 0; i < n; i++) grid.insertAdjacentHTML("beforeend", `<div class="skeleton"></div>`);
  $("#results").innerHTML = "";
  $("#results").appendChild(grid);
}

function describeSearch(body) {
  // Báo lên UI đang làm gì với option nào
  const steps = [];
  if (body.auto_expand) steps.push(`GPT mở rộng ${body.num_expansions || ""} query`.trim());
  const models = [];
  if (body.use_siglip !== false) models.push("SigLIP2");
  if (body.use_jina !== false) models.push("Jina");
  if (body.use_beit3 !== false) models.push("BEiT-3");
  if (body.use_bge !== false) models.push("BGE-caption");
  if (models.length) steps.push(`embed ${models.join("+")}`);
  if (body.use_rerank) steps.push(`BLIP2 rerank ${body.rerank_score_type || ""}`.trim());
  return steps.length ? ` (${steps.join(" → ")})` : "";
}

async function runSearch(path, body) {
  setStatus("⏳ Đang tìm kiếm..." + describeSearch(body));
  showSkeletons();
  $("#analysis").classList.add("hidden");
  try {
    const res = await api(path, body);
    state.results = res.results || [];
    state.baseResults = state.results.slice(); // bản gốc để filter object client-side
    state.page = 0;
    const t = typeof res.search_time === "number" ? ` (${res.search_time.toFixed(2)}s)` : "";
    setStatus(`${state.results.length} kết quả${t}` + (res.message ? ` — ${res.message}` : ""));
    // Output của GPT expand: các câu truy vấn đã rewrite (backend gắn vào từng kết quả)
    const exp = state.results[0]?.expanded_queries;
    state.expandedQueries = (Array.isArray(exp) && exp.length > 1) ? exp : null;
    const btn = $("#expand-info");
    btn.classList.toggle("hidden", !state.expandedQueries);
    if (state.expandedQueries) btn.textContent = `🔄 Query mở rộng (${state.expandedQueries.length})`;
    if (res.analysis) {
      $("#analysis").textContent = "🧠 " + (typeof res.analysis === "string" ? res.analysis : JSON.stringify(res.analysis, null, 1));
      $("#analysis").classList.remove("hidden");
    }
    updateObjectSuggestions();
    renderPage();
  } catch (err) {
    $("#results").innerHTML = "";
    setStatus(`Lỗi: ${err.message}`);
  }
}

// ============ modal chi tiết frame (OD + OCR + ASR) ============
function esRow(doc) {
  const t = doc.content ?? doc.text ?? JSON.stringify(doc);
  const s = doc.start_time, e = doc.end_time;
  const time = (s != null) ? `<small>[${fmtTime(+s)}${e != null ? " → " + fmtTime(+e) : ""}]</small> ` : "";
  return `<div class="row">${time}${t}</div>`;
}

function showDetailView(view) {
  const isVideo = view === "video";
  $("#detail-img").classList.toggle("hidden", isVideo);
  $("#detail-video").classList.toggle("hidden", !isVideo);
  document.querySelectorAll(".dtab").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  const vid = $("#detail-video");
  if (isVideo) {
    // nạp video 1 lần rồi tua tới đúng thời điểm của keyframe
    const want = `/video?name=${encodeURIComponent(state.detail?.video || "")}`;
    if (!vid.src.endsWith(want)) vid.src = want;
    const seek = () => { if (state.detail?.ts != null) vid.currentTime = Math.max(0, state.detail.ts - 1); };
    if (vid.readyState >= 1) seek(); else vid.addEventListener("loadedmetadata", seek, { once: true });
  } else {
    vid.pause();
  }
}
document.querySelectorAll(".dtab").forEach((b) =>
  b.addEventListener("click", () => showDetailView(b.dataset.view)));

async function loadNeighborShots(video, shotNumber) {
  const box = $("#detail-neighbors");
  if (!video || shotNumber == null) { box.innerHTML = ""; return; }
  box.innerHTML = `<span class="hint">Đang tải…</span>`;
  try {
    const from = Math.max(1, shotNumber - 4);
    const res = await api(`/videos/${encodeURIComponent(video)}/shots?start_shot=${from}&end_shot=${shotNumber + 4}`);
    const shots = (res.shots || []).slice().sort((a, b) => (a.shot_number ?? 0) - (b.shot_number ?? 0));
    if (!shots.length) { box.innerHTML = `<span class="hint">Không có shot lân cận.</span>`; return; }
    box.innerHTML = "";
    for (const s of shots) {
      // shot có positions {start, middle, end} — lấy frame đầu tiên có sẵn làm đại diện
      const kf = s.positions?.middle || s.positions?.start || s.positions?.end;
      const fp = kf?.file_path;
      if (!fp) continue;
      const num = s.shot_number;
      const el = document.createElement("div");
      el.className = "nb" + (num === shotNumber ? " current" : "");
      const ts = kf.timestamp_seconds;
      el.innerHTML = `<img loading="lazy" src="/files?p=${encodeURIComponent(fp)}">`
        + `<div>shot ${num ?? "?"}${ts != null ? " · " + fmtTime(ts) : ""}</div>`;
      el.addEventListener("click", () => openDetail({ file_path: fp, payload: kf.payload, score: null }));
      box.appendChild(el);
    }
  } catch (err) {
    box.innerHTML = `<span class="hint">Lỗi tải shot lân cận: ${err.message}</span>`;
  }
}

async function openDetail(item) {
  const m = itemMeta(item);
  const ov = $("#detail-overlay");
  ov.classList.remove("hidden");
  state.detail = { video: m.video, ts: m.ts };
  showDetailView("image");
  loadNeighborShots(m.video, item.payload?.shot?.number);
  $("#detail-img").src = m.filePath ? `/files?p=${encodeURIComponent(m.filePath)}` : "";
  $("#detail-raw").href = m.filePath ? `/files?p=${encodeURIComponent(m.filePath)}` : "#";
  $("#detail-title").textContent = `${m.video ?? "?"} — frame ${m.frame ?? "?"}`;
  $("#detail-meta").innerHTML = [
    m.ts != null ? `⏱ ${fmtTime(m.ts)} (${(+m.ts).toFixed(2)}s)` : "",
    item.payload?.shot_id ? `🎬 ${item.payload.shot_id}` : "",
    m.score != null ? `📈 score ${(+m.score).toFixed(4)}` : "",
    item.object_boost != null ? `📦 boost x${item.object_boost}` : "",
  ].filter(Boolean).join("<br>");
  $("#detail-bookmark").onclick = () => addBookmark(m);
  $("#detail-submit").onclick = () => submitToDres(m);

  // Object detection có sẵn trong payload — hiện ngay
  const renderOD = (od) => {
    if (!od || !od.object_count) return `<div class="none">Không phát hiện object nào.</div>`;
    return Object.entries(od.class_counts || {}).sort((a, b) => b[1] - a[1])
      .map(([c, n]) => `<div class="row">${c}: <b>${n}</b></div>`).join("")
      + `<div class="row">Tổng: <b>${od.object_count}</b> · conf max ${od.max_confidence ?? "?"}</div>`;
  };
  $("#detail-caption").innerHTML = m.caption
    ? `<div class="row">${m.caption}</div>`
    : `<div class="none">Chưa có caption (chạy offline/ingest_captions.py).</div>`;
  $("#detail-od").innerHTML = renderOD(item.payload?.object_detection);
  $("#detail-asr").innerHTML = `<div class="none">Đang tải…</div>`;
  $("#detail-ocr").innerHTML = `<div class="none">Đang tải…</div>`;

  // OCR/ASR (+ payload đầy đủ nếu card thiếu) từ backend
  try {
    const q = new URLSearchParams({ video: m.video ?? "" });
    if (m.frame != null) q.set("frame_index", m.frame);
    if (m.ts != null) q.set("ts", m.ts);
    const info = await api(`/frame/info?${q}`);
    const od2 = info.frame?.payload?.object_detection;
    if (od2 && !item.payload?.object_detection) $("#detail-od").innerHTML = renderOD(od2);
    const cap2 = info.frame?.payload?.caption;
    if (cap2 && !m.caption) $("#detail-caption").innerHTML = `<div class="row">${cap2}</div>`;
    $("#detail-asr").innerHTML = !info.asr_available
      ? `<div class="none">Elasticsearch ASR đang tắt.</div>`
      : (info.asr?.length ? info.asr.map(esRow).join("") : `<div class="none">Chưa có dữ liệu ASR cho video này (chưa ingest).</div>`);
    $("#detail-ocr").innerHTML = !info.ocr_available
      ? `<div class="none">Elasticsearch OCR đang tắt.</div>`
      : (info.ocr?.length ? info.ocr.map(esRow).join("") : `<div class="none">Chưa có dữ liệu OCR cho video này (chưa ingest).</div>`);
  } catch (err) {
    $("#detail-asr").innerHTML = $("#detail-ocr").innerHTML = `<div class="none">Lỗi tải: ${err.message}</div>`;
  }
}
function closeDetail() {
  $("#detail-overlay").classList.add("hidden");
  $("#detail-video").pause();
}
$("#detail-close").addEventListener("click", closeDetail);
$("#detail-overlay").addEventListener("click", (e) => { if (e.target.id === "detail-overlay") closeDetail(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });

// ============ filter object CLIENT-SIDE trên kết quả đang hiển thị ============
// Chỉnh filter sau khi search -> lọc/chấm lại ngay tập kết quả hiện tại,
// KHÔNG search lại (search lại sẽ ra ranking khác, mất ngữ cảnh đang xem).
// "person:2, car:1, dog" -> [{cls:"person",count:2},{cls:"car",count:1},{cls:"dog",count:null}]
// Số trần ("10") hoặc "tổng:10"/"total:10" -> lọc theo TỔNG số object (cls rỗng).
function parseObjectSpecs(text) {
  return (text || "").split(",").map((s) => s.trim()).filter(Boolean).map((part) => {
    if (/^\d+$/.test(part)) return { cls: "", count: +part };
    const m = part.match(/^(.+?)\s*[:=]\s*(\d+)$/);
    if (!m) return { cls: part.toLowerCase(), count: null };
    const cls = m[1].trim().toLowerCase();
    return { cls: (cls === "tổng" || cls === "tong" || cls === "total") ? "" : cls, count: +m[2] };
  });
}

function applyObjectFilter() {
  if (!state.baseResults) return;
  const f = $("#form-kis");
  const tol = +f.object_tolerance.value || 0;
  const mode = f.object_mode.value;
  const floor = Math.min(1, Math.max(0, +f.object_floor.value || 0.25));
  const specs = parseObjectSpecs(f.object_class.value);

  const actualOf = (r, cls) => {
    const od = r.payload?.object_detection || {};
    return cls ? (od.class_counts?.[cls] || 0) : (od.object_count || 0);
  };

  if (!specs.length) {
    state.results = state.baseResults.slice(); // không filter -> trả về bản gốc
  } else if (mode === "soft") {
    const scale = Math.max(1, tol);
    state.results = state.baseResults
      .map((r) => {
        const mults = specs.map((sp) => {
          const actual = actualOf(r, sp.cls);
          if (sp.count == null) return actual >= 1 ? 1 : floor;
          return floor + (1 - floor) * Math.exp(-Math.abs(actual - sp.count) / scale);
        });
        const mult = mults.reduce((a, b) => a + b, 0) / mults.length; // trung bình, luôn trong [floor,1]
        const actuals = Object.fromEntries(specs.map((sp) => [sp.cls || "tổng", actualOf(r, sp.cls)]));
        return { ...r, score: (r.score ?? 0) * mult, object_boost: +mult.toFixed(4), object_actual: actuals };
      })
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  } else { // hard: phải thoả MỌI spec
    state.results = state.baseResults.filter((r) => specs.every((sp) => {
      const actual = actualOf(r, sp.cls);
      return sp.count == null ? actual >= 1 : Math.abs(actual - sp.count) <= tol;
    }));
  }
  state.page = 0;
  const desc = specs.map((sp) => (sp.cls || "tổng") + (sp.count != null ? `=${sp.count}±${tol}` : "≥1")).join(", ");
  setStatus(`${state.results.length}/${state.baseResults.length} kết quả sau lọc object`
    + (desc ? ` — ${desc} (${mode})` : ""));
  renderPage();
}
["object_class", "object_tolerance", "object_mode", "object_floor"].forEach((name) => {
  const el = $("#form-kis")[name];
  el.addEventListener("input", applyObjectFilter);
  el.addEventListener("change", applyObjectFilter);
});

// ============ gợi ý object class từ kết quả trả về ============
function updateObjectSuggestions() {
  const box = $("#obj-suggest");
  const agg = {}; // class -> số ảnh chứa class đó
  for (const r of state.baseResults || state.results) {
    const cc = r.payload?.object_detection?.class_counts || {};
    for (const cls of Object.keys(cc)) agg[cls] = (agg[cls] || 0) + 1;
  }
  const top = Object.entries(agg).sort((a, b) => b[1] - a[1]).slice(0, 14);
  if (!top.length) { box.classList.add("hidden"); box.innerHTML = ""; return; }
  box.innerHTML = "Trong kết quả có: " + top.map(([cls, n]) =>
    `<button type="button" class="obj-chip" data-cls="${cls}">${cls} <small>(${n})</small></button>`).join(" ");
  box.classList.remove("hidden");
  const selected = new Set(parseObjectSpecs($("#form-kis").object_class.value).map((s) => s.cls));
  box.querySelectorAll(".obj-chip").forEach((b) => {
    if (selected.has(b.dataset.cls)) b.classList.add("on");
    b.addEventListener("click", () => {
      // click = thêm/bớt class khỏi danh sách (lọc được NHIỀU object cùng lúc)
      const inp = $("#form-kis").object_class;
      const specs = parseObjectSpecs(inp.value);
      const idx = specs.findIndex((s) => s.cls === b.dataset.cls);
      if (idx >= 0) specs.splice(idx, 1); else specs.push({ cls: b.dataset.cls, count: null });
      inp.value = specs.map((s) => s.cls + (s.count != null ? ":" + s.count : "")).join(", ");
      b.classList.toggle("on", idx < 0);
      applyObjectFilter(); // lọc ngay trên kết quả đang hiển thị
    });
  });
}

// ============ nút xem query GPT đã mở rộng ============
$("#expand-info").addEventListener("click", () => {
  const panel = $("#analysis");
  const showing = !panel.classList.contains("hidden") && panel.dataset.mode === "expand";
  if (showing) { panel.classList.add("hidden"); panel.dataset.mode = ""; return; }
  panel.textContent = "🔄 Các query GPT đã mở rộng (câu đầu là bản dịch trực tiếp):\n  • "
    + (state.expandedQueries || []).join("\n  • ");
  panel.dataset.mode = "expand";
  panel.classList.remove("hidden");
});

// ============ toggle sidebar cấu hình ============
function setOptsHidden(hidden) {
  document.body.classList.toggle("opts-hidden", hidden);
  localStorage.setItem("opts_hidden", hidden ? "1" : "");
}
$("#opts-close").addEventListener("click", () => setOptsHidden(true));
$("#opts-toggle").addEventListener("click", () => setOptsHidden(!document.body.classList.contains("opts-hidden")));
if (localStorage.getItem("opts_hidden") === "1") setOptsHidden(true);

// Gộp kết quả: mỗi shot (3 keyframe start/middle/end) hoặc mỗi video còn 1 ô đại
// diện — giữ kết quả điểm cao nhất, đếm số keyframe đã gộp để hiện badge.
function groupResults(results, mode) {
  if (mode === "none") return results;
  const seen = new Map();
  const out = [];
  for (const r of results) {
    const p = r.payload || {};
    const key = mode === "video"
      ? (p.video?.name || null)
      : (p.shot_id || (p.video?.name && p.shot?.number != null ? `${p.video.name}#${p.shot.number}` : null));
    if (!key) { out.push(r); continue; }
    const cur = seen.get(key);
    if (!cur) {
      const copy = { ...r, _groupCount: 1 };
      seen.set(key, copy);
      out.push(copy);
    } else {
      cur._groupCount++;
      if ((r.score ?? 0) > (cur.score ?? 0)) Object.assign(cur, r, { _groupCount: cur._groupCount });
    }
  }
  return out;
}

function renderPage() {
  const displayed = groupResults(state.results, $("#group-mode").value);
  const total = displayed.length;
  if (!total) {
    $("#results").innerHTML = `<div id="empty-state"><div class="empty-icon">🤷</div><p>Không có kết quả phù hợp.</p><p class="hint">Thử hạ ngưỡng điểm hoặc đổi mô tả.</p></div>`;
    $("#pager").style.visibility = "hidden";
    return;
  }
  const pages = Math.max(1, Math.ceil(displayed.length / PAGE_SIZE));
  state.page = Math.min(state.page, pages - 1);
  $("#page-info").textContent = `${state.page + 1}/${pages}`;
  $("#pager").style.visibility = pages > 1 ? "visible" : "hidden";

  const grid = document.createElement("div");
  grid.className = "grid";
  const slice = displayed.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
  for (const item of slice) {
    const seq = item.steps || item.sequence || item.events || item.frames;
    if (Array.isArray(seq)) {
      const wrap = document.createElement("div");
      wrap.className = "sequence";
      const head = document.createElement("div");
      head.className = "seq-head";
      const m = itemMeta(item);
      head.innerHTML = `<b>${item.label ?? m.video ?? ""}</b><br>${item.label ? m.video : seq.length + " events"}<br>${m.score != null ? "score " + (+m.score).toFixed(2) : ""}`;
      wrap.appendChild(head);
      wrap.append(...seq.map((s) => card(s)));
      grid.appendChild(wrap);
    } else {
      grid.appendChild(card(item));
    }
  }
  $("#results").innerHTML = "";
  $("#results").appendChild(grid);
  window.scrollTo({ top: 0, behavior: "smooth" });
}
$("#group-mode").addEventListener("change", () => { state.page = 0; renderPage(); });
$("#prev-page").addEventListener("click", () => { if (state.page > 0) { state.page--; renderPage(); } });
$("#next-page").addEventListener("click", () => { state.page++; renderPage(); });

function card(item) {
  const m = itemMeta(item);
  const el = document.createElement("div");
  el.className = "card";

  const thumb = document.createElement("div");
  thumb.className = "thumb";
  if (m.filePath) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/files?p=${encodeURIComponent(m.filePath)}`;
    img.addEventListener("error", () => (img.style.display = "none"));
    thumb.appendChild(img);
    if (m.gifPath) {
      const gif = document.createElement("img");
      gif.className = "gif";
      thumb.addEventListener("mouseenter", () => { if (!gif.src) gif.src = `/files?p=${encodeURIComponent(m.gifPath)}`; });
      thumb.appendChild(gif);
    }
    thumb.addEventListener("click", (e) => {
      if (e.target.tagName !== "BUTTON") openDetail(item);
    });
  }

  // badges đè lên ảnh (kiểu YouTube)
  if (m.video) thumb.insertAdjacentHTML("beforeend", `<span class="badge video">${m.video}</span>`);
  const timeTxt = [m.frame != null ? "f" + m.frame : "", m.ts != null ? fmtTime(m.ts) : ""].filter(Boolean).join(" · ");
  if (timeTxt) thumb.insertAdjacentHTML("beforeend", `<span class="badge time">${timeTxt}</span>`);
  if (m.score != null) thumb.insertAdjacentHTML("beforeend", `<span class="badge score">${(+m.score).toFixed(3)}</span>`);
  if (item._groupCount > 1) {
    const unit = $("#group-mode").value === "video" ? "video này" : "shot này";
    thumb.insertAdjacentHTML("beforeend", `<span class="badge frames" title="${item._groupCount} keyframe khớp trong ${unit}">🎞 ${item._groupCount}</span>`);
  }
  const od = item.payload?.object_detection;
  if (od && od.object_count > 0) {
    const detail = Object.entries(od.class_counts || {}).map(([c, n]) => `${c}: ${n}`).join(", ");
    thumb.insertAdjacentHTML("beforeend", `<span class="badge objects" title="${detail}">📦 ${od.object_count}</span>`);
  }

  // actions hiện khi hover
  const actions = document.createElement("div");
  actions.className = "hover-actions";
  const bm = document.createElement("button");
  const isBm = state.bookmarks.some((b) => b.video === m.video && b.frame === m.frame);
  bm.textContent = isBm ? "★" : "⭐";
  if (isBm) bm.classList.add("bookmarked");
  bm.title = "Bookmark (dùng cho TRAKE)";
  bm.addEventListener("click", (e) => { e.stopPropagation(); addBookmark(m); bm.textContent = "★"; bm.classList.add("bookmarked"); });
  const sub = document.createElement("button");
  sub.textContent = "🚀 Nộp";
  sub.title = "Nộp DRES theo task type đang chọn";
  sub.addEventListener("click", (e) => { e.stopPropagation(); submitToDres(m); });
  actions.append(bm, sub);
  // Ở chế độ agent: nói cho trợ lý biết ảnh nào giống nhất để nó tìm sát hơn
  if (!$("#agent-panel").classList.contains("hidden")) {
    const pick = document.createElement("button");
    pick.textContent = "🤖 Giống";
    pick.title = "Báo cho agent: ảnh này giống cảnh cần tìm";
    pick.addEventListener("click", (e) => {
      e.stopPropagation();
      sendAgent(`Ảnh này giống cảnh tôi cần: video ${m.video}, khoảng ${m.ts?.toFixed?.(1) ?? "?"}s`
        + `. Hãy tìm quanh đó hoặc lọc trong video này.`);
    });
    actions.append(pick);
  }
  thumb.appendChild(actions);
  el.appendChild(thumb);

  // Caption do BLIP2 sinh
  if (m.caption) {
    const cap = document.createElement("div");
    cap.className = "card-caption";
    cap.textContent = m.caption;
    cap.title = m.caption;
    el.appendChild(cap);
  }

  // Dòng số lượng object: class đang lọc hiện trước và tô đậm
  if (od && od.object_count > 0) {
    const filtered = new Set(parseObjectSpecs($("#form-kis").object_class.value).map((s) => s.cls).filter(Boolean));
    const entries = Object.entries(od.class_counts || {})
      .sort((a, b) => (filtered.has(b[0]) - filtered.has(a[0])) || b[1] - a[1]);
    const shown = entries.slice(0, 4).map(([c, n]) =>
      `<span class="obj-count${filtered.has(c) ? " hit" : ""}">${c} <b>${n}</b></span>`).join("");
    const rest = entries.length - 4;
    el.insertAdjacentHTML("beforeend",
      `<div class="obj-counts" title="${entries.map(([c, n]) => c + ": " + n).join(", ")}">`
      + shown + (rest > 0 ? `<span class="obj-count">+${rest}</span>` : "") + `</div>`);
  }

  if (m.text) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `<div class="snippet"></div>`;
    meta.querySelector(".snippet").textContent = m.text.slice(0, 260);
    el.appendChild(meta);
  }
  return el;
}

// ============ bookmarks ============
function persistBookmarks() {
  localStorage.setItem("bookmarks", JSON.stringify(state.bookmarks));
  renderBookmarks();
}
function addBookmark(m) {
  if (!m.video || m.frame == null) return setStatus("Thiếu video/frame, không bookmark được.");
  if (state.bookmarks.some((b) => b.video === m.video && b.frame === m.frame)) return;
  state.bookmarks.push({ video: m.video, frame: m.frame, ts: m.ts });
  persistBookmarks();
}
function renderBookmarks() {
  $("#bm-count").textContent = state.bookmarks.length ? `(${state.bookmarks.length})` : "";
  $("#rail-bm-count").textContent = state.bookmarks.length;
  $("#bookmarks").innerHTML = state.bookmarks.map((b, i) =>
    `<div class="bm"><span>${i + 1}. ${b.video} · f${b.frame} · ${fmtTime(b.ts)}</span><button data-i="${i}">×</button></div>`).join("");
  document.querySelectorAll("#bookmarks button").forEach((btn) =>
    btn.addEventListener("click", () => { state.bookmarks.splice(+btn.dataset.i, 1); persistBookmarks(); }));
}
$("#clear-bookmarks").addEventListener("click", () => { state.bookmarks = []; persistBookmarks(); });
renderBookmarks();

// ============ DRES ============
function renderDres() {
  const logged = !!state.dres?.sessionId;
  $("#dres-login").classList.toggle("hidden", logged);
  $("#dres-session").classList.toggle("hidden", !logged);
  const pill = $("#dres-status");
  pill.textContent = logged ? `DRES: ${state.dres.username} ✔` : "DRES: chưa đăng nhập";
  pill.className = `pill ${logged ? "pill-on" : "pill-off"}`;
  if (logged) loadEvaluations();
}
function persistDres() { localStorage.setItem("dres", JSON.stringify(state.dres)); renderDres(); }

$("#dres-login-btn").addEventListener("click", async (e) => {
  e.preventDefault();
  const r = await dres("login", { username: $("#dres-user").value, password: $("#dres-pass").value });
  if (!r.ok || !r.data.sessionId) return setStatus(`DRES login lỗi: ${r.data.description || r.status}`);
  state.dres = { sessionId: r.data.sessionId, username: r.data.username, evaluationId: null };
  persistDres();
});
$("#dres-logout-btn").addEventListener("click", (e) => { e.preventDefault(); state.dres = null; persistDres(); });

async function loadEvaluations() {
  const r = await dres(`client/evaluation/list?session=${encodeURIComponent(state.dres.sessionId)}`);
  if (!r.ok || !Array.isArray(r.data)) return;
  $("#dres-eval").innerHTML = r.data.map((ev) =>
    `<option value="${ev.id}" ${ev.id === state.dres.evaluationId ? "selected" : ""}>${ev.name}</option>`).join("");
  if (!state.dres.evaluationId && r.data[0]) { state.dres.evaluationId = r.data[0].id; persistDres(); }
}
$("#dres-refresh-evals").addEventListener("click", (e) => { e.preventDefault(); loadEvaluations(); });
$("#dres-eval").addEventListener("change", (e) => { state.dres.evaluationId = e.target.value; persistDres(); });
$("#dres-eval-manual").addEventListener("change", (e) => {
  if (e.target.value.trim()) { state.dres.evaluationId = e.target.value.trim(); persistDres(); }
});
$("#task-type").addEventListener("change", (e) =>
  $("#qa-answer").classList.toggle("hidden", e.target.value !== "QA"));

$("#dres-state-btn").addEventListener("click", async (e) => {
  e.preventDefault();
  if (!state.dres?.evaluationId) return setStatus("Chưa chọn evaluation.");
  const s = encodeURIComponent(state.dres.sessionId);
  const ev = encodeURIComponent(state.dres.evaluationId);
  const [st, task] = await Promise.all([
    dres(`client/evaluation/${ev}/state?session=${s}`),
    dres(`client/evaluation/${ev}/currentTask?session=${s}`),
  ]);
  $("#dres-info").textContent =
    `Trạng thái: ${JSON.stringify(st.data).slice(0, 160)}\nTask: ${JSON.stringify(task.data).slice(0, 160)}`;
});

function persistSubmitted() {
  localStorage.setItem("submitted", JSON.stringify(state.submitted));
  $("#submitted").innerHTML = state.submitted.length
    ? `Đã nộp ${state.submitted.length}:<br>` + state.submitted.slice(-5).map((s) => `${s.ok ? "✅" : "❌"} ${s.key}`).join("<br>")
    : "";
}
persistSubmitted();

// Format nộp DRES v2 (AIC 2025; DRES_URL đổi được qua env khi 2026 công bố):
// KIS: {mediaItemName, start(ms), end(ms)} | QA: "QA-<ans>-<video>-<ms>" | TRAKE: "TR-<video>-<f1>,<f2>,..."
async function dresSubmit(answers, key) {
  if (!state.dres?.sessionId) return setStatus("Chưa đăng nhập DRES.");
  if (!state.dres.evaluationId) return setStatus("Chưa chọn evaluation.");
  if (state.submitted.some((s) => s.key === key)) return setStatus("Đáp án này đã nộp rồi — không nộp trùng!");
  const r = await dres(`submit/${encodeURIComponent(state.dres.evaluationId)}?session=${encodeURIComponent(state.dres.sessionId)}`,
    { answerSets: [{ answers }] });
  state.submitted.push({ key, time: Date.now(), ok: r.ok });
  persistSubmitted();
  setStatus(r.ok ? `✅ Đã nộp (${r.data.submission || r.data.status || r.status})` : `❌ Nộp thất bại: ${r.data.description || r.status}`);
}

const SUBMIT_WINDOW_S = 2; // KIS: nộp [ts-2s, ts+2s] quanh keyframe

function submitToDres(m) {
  const taskType = $("#task-type").value;
  if (!m.video) return setStatus("Kết quả thiếu tên video.");
  if (taskType === "KIS") {
    if (m.ts == null) return setStatus("Kết quả thiếu timestamp.");
    const start = Math.max(0, Math.round((m.ts - SUBMIT_WINDOW_S) * 1000));
    const end = Math.round((m.ts + SUBMIT_WINDOW_S) * 1000);
    dresSubmit([{ mediaItemName: String(m.video), start, end }], `KIS:${m.video}:${start}:${end}`);
  } else if (taskType === "QA") {
    const ans = $("#qa-answer").value.trim();
    if (!ans) return setStatus("Nhập câu trả lời QA.");
    const ms = Math.round((m.ts ?? 0) * 1000);
    dresSubmit([{ text: `QA-${ans}-${m.video}-${ms}` }], `QA:${ans}:${m.video}:${ms}`);
  } else {
    setStatus("TRAKE: bookmark các frame theo thứ tự rồi bấm 🚀 TRAKE ở panel Bookmarks.");
  }
}

$("#submit-trake").addEventListener("click", (e) => {
  e.preventDefault();
  if (!state.bookmarks.length) return setStatus("Chưa có bookmark nào.");
  const video = state.bookmarks[0].video;
  if (!state.bookmarks.every((b) => b.video === video))
    return setStatus("TRAKE cần tất cả frame cùng một video.");
  const frames = state.bookmarks.map((b) => b.frame).join(",");
  dresSubmit([{ text: `TR-${video}-${frames}` }], `TRAKE:${video}:${frames}`);
});

renderDres();

// ============ drawer DRES & Bookmarks ============
const toggleRail = (open) => $("#rail").classList.toggle("open", open);
$("#toggle-rail").addEventListener("click", () => toggleRail(!$("#rail").classList.contains("open")));
$("#dres-status").addEventListener("click", () => toggleRail(true));
$("#close-rail").addEventListener("click", () => toggleRail(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") toggleRail(false); });

// ============================================================
// CHẾ ĐỘ AGENT: trợ lý tự gọi tool search, hỏi lại, cập nhật kết quả
// ============================================================
const agent = { messages: [], busy: false };

function setMode(mode) {
  const isAgent = mode === "agent";
  $("#agent-panel").classList.toggle("hidden", !isAgent);
  document.querySelector(".query-zone").classList.toggle("hidden", isAgent);
  document.body.classList.toggle("opts-hidden", isAgent); // agent tự quyết option
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  localStorage.setItem("ui_mode", mode);
  if (isAgent && !agent.messages.length) {
    agentLog("bot", "Mô tả cảnh bạn cần tìm. Tôi sẽ tự tìm, hỏi lại nếu chưa rõ, và lọc dần cho tới khi ra ứng viên tốt.");
  }
}
document.querySelectorAll(".mode-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

function agentLog(kind, text) {
  const el = document.createElement("div");
  el.className = `ag-msg ${kind}`;
  el.textContent = text;
  $("#agent-log").appendChild(el);
  $("#agent-log").scrollTop = $("#agent-log").scrollHeight;
  return el;
}

function agentSuggestions(list) {
  const box = $("#agent-suggest");
  box.innerHTML = "";
  for (const s of list || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s;
    b.addEventListener("click", () => { box.innerHTML = ""; sendAgent(s); });
    box.appendChild(b);
  }
}

async function sendAgent(text) {
  if (agent.busy || !text.trim()) return;
  agent.busy = true;
  $("#agent-suggest").innerHTML = "";
  agentLog("user", text);
  agent.messages.push({ role: "user", content: text });
  const typing = agentLog("tool", "⏳ đang tìm…");
  try {
    const res = await api("/agent/chat", { messages: agent.messages });
    typing.remove();
    for (const a of res.actions || []) {
      const argTxt = a.args?.query || a.args?.question || JSON.stringify(a.args || {});
      agentLog("tool", `🔧 ${a.tool}(${String(argTxt).slice(0, 60)}) → ${a.summary}`);
    }
    if (res.message) agentLog("bot", res.message);
    agent.messages = res.messages || agent.messages;
    if (res.results?.length) {
      state.results = res.results;
      state.baseResults = res.results.slice();
      state.page = 0;
      setStatus(`${res.results.length} kết quả (agent)`);
      updateObjectSuggestions();
      renderPage();
    }
    if (res.question?.suggestions?.length) agentSuggestions(res.question.suggestions);
  } catch (err) {
    typing.remove();
    agentLog("err", `Lỗi: ${err.message}`);
  } finally {
    agent.busy = false;
  }
}

$("#agent-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const ta = e.target.msg;
  const text = ta.value.trim();
  ta.value = "";
  sendAgent(text);
});
$("#agent-reset").addEventListener("click", () => {
  agent.messages = [];
  $("#agent-log").innerHTML = "";
  $("#agent-suggest").innerHTML = "";
  agentLog("bot", "Hội thoại mới. Mô tả cảnh bạn cần tìm nhé.");
});

setMode(localStorage.getItem("ui_mode") || "manual");
