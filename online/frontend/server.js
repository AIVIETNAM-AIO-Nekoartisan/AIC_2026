// VBS web frontend: static UI + proxy sang FastAPI backend + DRES.
// Node >= 18 (dùng global fetch). Chỉ 1 dependency: express.
const path = require("path");
const fs = require("fs");
const express = require("express");

const PORT = parseInt(process.env.PORT || "3000", 10);
const API_URL = process.env.API_URL || "http://localhost:8000";
const DRES_URL = process.env.DRES_URL || "https://eventretrieval.oj.io.vn";
// Thư mục gốc chứa keyframes (results/...). Mặc định: repo root (cha của web/).
const DATA_ROOT = path.resolve(process.env.DATA_ROOT || path.join(__dirname, ".."));

const app = express();
app.use(express.json({ limit: "50mb" })); // image_base64 có thể lớn
app.use(express.static(path.join(__dirname, "public")));

// ---- Serve keyframe images theo file_path trong payload Qdrant ----
// ponytail: payload có thể chứa path tuyệt đối của máy ingest; remap mọi thứ
// từ "results/" trở đi vào DATA_ROOT. Nếu cấu trúc dữ liệu đổi, thêm prefix map.
function resolveDataPath(p) {
  if (!p) return null;
  const candidates = [];
  const norm = p.replace(/\\/g, "/");
  const idx = norm.indexOf("results/");
  if (idx >= 0) candidates.push(path.join(DATA_ROOT, norm.slice(idx)));
  candidates.push(path.resolve(DATA_ROOT, norm)); // path tương đối so với DATA_ROOT
  candidates.push(path.resolve(norm)); // path tuyệt đối (chạy không Docker)
  for (const c of candidates) {
    const resolved = path.resolve(c);
    if (!resolved.startsWith(DATA_ROOT + path.sep) && resolved !== DATA_ROOT) continue; // chặn path traversal
    if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) return resolved;
  }
  return null;
}

app.get("/files", (req, res) => {
  const resolved = resolveDataPath(String(req.query.p || ""));
  if (!resolved) return res.status(404).end();
  res.sendFile(resolved);
});

// ---- Serve video gốc theo tên (vd L27_V001) để xem/tua trong modal ----
// res.sendFile hỗ trợ Range nên <video> tua được. Chỉ nhận [A-Za-z0-9_-].
const VIDEO_DIRS = (process.env.VIDEO_DIRS || "data/video,data/videos,videos").split(",");
app.get("/video", (req, res) => {
  const name = String(req.query.name || "").replace(/\.mp4$/i, "");
  if (!/^[\w-]+$/.test(name)) return res.status(400).end();
  for (const dir of VIDEO_DIRS) {
    for (const ext of [".mp4", ".mkv", ".webm", ".mov"]) {
      const resolved = resolveDataPath(path.join(dir.trim(), name + ext));
      if (resolved) return res.sendFile(resolved);
    }
  }
  res.status(404).end();
});

// ---- Proxy JSON sang FastAPI backend (tránh CORS, 1 origin duy nhất) ----
app.all(/^\/api\/(.*)/, async (req, res) => {
  const target = `${API_URL}/${req.params[0]}${req.url.includes("?") ? req.url.slice(req.url.indexOf("?")) : ""}`;
  try {
    const init = { method: req.method, headers: { "Content-Type": "application/json" } };
    if (req.method !== "GET" && req.method !== "HEAD") init.body = JSON.stringify(req.body);
    const r = await fetch(target, init);
    res.status(r.status).type("application/json").send(await r.text());
  } catch (e) {
    res.status(502).json({ success: false, message: `Backend không phản hồi: ${e.message}` });
  }
});

// ---- Proxy DRES (browser không gọi thẳng được vì CORS) ----
// Passthrough: /dres/<path> -> {DRES_URL}/api/v2/<path> (giữ nguyên query + body)
app.all(/^\/dres\/(.*)/, async (req, res) => {
  const qs = req.url.includes("?") ? req.url.slice(req.url.indexOf("?")) : "";
  const target = `${DRES_URL}/api/v2/${req.params[0]}${qs}`;
  try {
    const init = { method: req.method, headers: { "Content-Type": "application/json" } };
    if (req.method !== "GET" && req.method !== "HEAD") init.body = JSON.stringify(req.body);
    const r = await fetch(target, init);
    res.status(r.status).type("application/json").send(await r.text());
  } catch (e) {
    res.status(502).json({ error: `DRES không phản hồi: ${e.message}` });
  }
});

app.listen(PORT, () => {
  console.log(`VBS web UI: http://localhost:${PORT}`);
  console.log(`  API_URL=${API_URL}  DRES_URL=${DRES_URL}  DATA_ROOT=${DATA_ROOT}`);
});
