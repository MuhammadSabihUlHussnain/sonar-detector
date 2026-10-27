// Sonar Object Detector: runs a YOLO26s ONNX model in the browser with onnxruntime-web.
// The model is exported with NMS built in: output [1, 300, 6] = x1, y1, x2, y2, score, class
// in letterboxed input pixels.

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
  "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000",
  "#ffd8b1", "#00a3ff", "#ff7f7f", "#7fff7f", "#c0c0c0", "#ff9900", "#66ccff", "#cc99ff"];

const el = (id) => document.getElementById(id);
const canvas = el("canvas");
const ctx = canvas.getContext("2d");

let meta = null;
let session = null;
let current = null;   // { img, detections, gt, name }

async function init() {
  try {
    meta = await (await fetch("model/meta.json")).json();
    session = await ort.InferenceSession.create(meta.model, { executionProviders: ["wasm"] });
    setStatus("Model ready");
  } catch (e) {
    setStatus("Could not load the model: " + e.message);
    return;
  }
  renderModelInfo();
  loadSamples();
}

function setStatus(text) { el("status").textContent = text; }

// ---------- input ----------

const drop = el("drop");
el("file").addEventListener("change", (e) => e.target.files[0] && openFile(e.target.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el("file").click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && openFile(e.dataTransfer.files[0]));

el("conf").addEventListener("input", () => { el("conf-out").textContent = Number(el("conf").value).toFixed(2); draw(); });
el("show-gt").addEventListener("change", draw);

function openFile(file) {
  if (!file.type.startsWith("image/")) { setStatus("That file is not an image."); return; }
  const url = URL.createObjectURL(file);
  document.querySelectorAll(".sample.active").forEach((b) => b.classList.remove("active"));
  loadImage(url, null, file.name);
}

function loadImage(src, gt, name) {
  const img = new Image();
  img.onload = () => detect(img, gt, name);
  img.onerror = () => setStatus("Could not read that image.");
  img.src = src;
}

// ---------- inference ----------

async function detect(img, gt, name) {
  if (!session) return;
  setStatus("Detecting…");
  const size = meta.imgsz;
  const w = img.naturalWidth, h = img.naturalHeight;
  const scale = Math.min(size / w, size / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const padX = (size - nw) / 2, padY = (size - nh) / 2;

  // letterbox onto a grey square, as in training
  const off = document.createElement("canvas");
  off.width = size; off.height = size;
  const octx = off.getContext("2d");
  octx.fillStyle = "rgb(114,114,114)";
  octx.fillRect(0, 0, size, size);
  octx.drawImage(img, padX, padY, nw, nh);
  const px = octx.getImageData(0, 0, size, size).data;
  const area = size * size;
  const input = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    input[i] = px[i * 4] / 255;
    input[i + area] = px[i * 4 + 1] / 255;
    input[i + 2 * area] = px[i * 4 + 2] / 255;
  }

  const t0 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, size, size]) });
  const ms = performance.now() - t0;
  const data = out[session.outputNames[0]].data;

  const dets = [];
  for (let i = 0; i < data.length; i += 6) {
    const score = data[i + 4];
    if (score < 0.05) continue;
    dets.push({
      x1: clamp((data[i] - padX) / scale, 0, w), y1: clamp((data[i + 1] - padY) / scale, 0, h),
      x2: clamp((data[i + 2] - padX) / scale, 0, w), y2: clamp((data[i + 3] - padY) / scale, 0, h),
      score, cls: Math.round(data[i + 5]),
    });
  }
  dets.sort((a, b) => b.score - a.score);
  current = { img, detections: dets, gt, name };
  setStatus(`${name} · ${w}×${h}px · ${ms.toFixed(0)} ms`);
  draw();
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ---------- drawing ----------

function draw() {
  if (!current) return;
  const { img, detections, gt } = current;
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w; canvas.height = h;
  el("empty").hidden = true;
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(w, h) / 400));
  const font = Math.max(12, Math.round(Math.max(w, h) / 55));
  ctx.font = `600 ${font}px system-ui, sans-serif`;

  if (gt && el("show-gt").checked) {
    ctx.setLineDash([lw * 3, lw * 2]);
    ctx.lineWidth = lw;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    for (const [, x1, y1, x2, y2] of gt) ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  }

  const conf = Number(el("conf").value);
  const shown = detections.filter((d) => d.score >= conf);
  shown.forEach((d, i) => {
    const color = PALETTE[d.cls % PALETTE.length];
    ctx.lineWidth = lw;
    ctx.strokeStyle = color;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const label = `${i + 1} ${meta.classes[d.cls]} ${(d.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 8;
    const ty = d.y1 - font - 6 < 0 ? d.y1 : d.y1 - font - 6;
    ctx.fillStyle = color;
    ctx.fillRect(d.x1, ty, tw, font + 6);
    ctx.fillStyle = "#000";
    ctx.fillText(label, d.x1 + 4, ty + font);
  });
  renderTable(shown, gt);
}

function quality(cls) {
  const m = meta.metrics?.per_class?.[meta.classes[cls]];
  if (!m) return { text: "not measured", cls: "" };
  const v = m.map50;
  const word = v >= 0.6 ? "good" : v >= 0.3 ? "fair" : "weak";
  return { text: `${word} (mAP50 ${v.toFixed(2)})`, cls: v >= 0.6 ? "q-good" : v >= 0.3 ? "q-mid" : "q-bad" };
}

function renderTable(shown, gt) {
  const tbody = el("det-table").querySelector("tbody");
  tbody.innerHTML = "";
  const counts = {};
  shown.forEach((d, i) => {
    const name = meta.classes[d.cls];
    counts[name] = (counts[name] || 0) + 1;
    const q = quality(d.cls);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td><span class="swatch" style="background:${PALETTE[d.cls % PALETTE.length]}"></span>${name}</td>` +
      `<td>${(d.score * 100).toFixed(1)}%</td><td class="${q.cls}">${q.text}</td>`;
    tbody.appendChild(tr);
  });
  el("det-table").hidden = shown.length === 0;
  const parts = Object.entries(counts).map(([k, v]) => `${v} ${k}`);
  let text = shown.length ? `Found ${shown.length} object${shown.length > 1 ? "s" : ""}: ${parts.join(", ")}.`
    : "No objects above the confidence threshold. Try lowering it.";
  if (gt && el("show-gt").checked) {
    const names = gt.map((g) => g[0]);
    text += ` True labels: ${gt.length ? countList(names) : "none (empty image)"}.`;
  }
  el("summary").textContent = text;
}

function countList(names) {
  const c = {};
  names.forEach((n) => (c[n] = (c[n] || 0) + 1));
  return Object.entries(c).map(([k, v]) => `${v} ${k}`).join(", ");
}

// ---------- model info and samples ----------

function renderModelInfo() {
  const m = meta.metrics || {};
  const o = m.overall || {};
  el("n-classes").textContent = meta.classes.length;
  el("model-blurb").textContent =
    `${meta.arch || "YOLO26n"} (${meta.params_m}M parameters), input ${meta.imgsz}px, trained on ${meta.train_images.toLocaleString()} sonar images ` +
    `for ${meta.epochs} epochs. On ${(o.images || 0).toLocaleString()} held-out test images it scores mAP50 ${fmt(o.map50)} ` +
    `(precision ${fmt(o.precision)}, recall ${fmt(o.recall)}). ${meta.note || ""}`;
  const tbody = el("metric-table").querySelector("tbody");
  Object.entries(m.per_class || {}).sort((a, b) => b[1].map50 - a[1].map50).forEach(([name, v]) => {
    const tr = document.createElement("tr");
    const q = v.map50 >= 0.6 ? "q-good" : v.map50 >= 0.3 ? "q-mid" : "q-bad";
    tr.innerHTML = `<td>${name}</td><td class="${q}">${fmt(v.map50)}</td><td>${fmt(v.precision)}</td><td>${fmt(v.recall)}</td><td>${v.boxes ?? ""}</td>`;
    tbody.appendChild(tr);
  });
}

const fmt = (v) => (typeof v === "number" ? v.toFixed(2) : "–");

async function loadSamples() {
  let list = [];
  try { list = await (await fetch("samples/samples.json")).json(); } catch { return; }
  const box = el("sample-list");
  list.forEach((s) => {
    const b = document.createElement("button");
    b.className = "sample";
    b.type = "button";
    b.title = `${s.label} (${s.source})`;
    b.innerHTML = `<img src="samples/${s.file}" alt="" loading="lazy"><span>${s.label}</span>`;
    b.addEventListener("click", () => {
      document.querySelectorAll(".sample.active").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      loadImage(`samples/${s.file}`, s.boxes, `${s.label} (${s.source})`);
    });
    box.appendChild(b);
  });
  // ?sample=N opens that sample straight away, ?conf=0.4 sets the threshold (handy for sharing)
  const q = new URLSearchParams(location.search);
  const c = Number(q.get("conf"));
  if (c > 0 && c < 1) { el("conf").value = c; el("conf-out").textContent = c.toFixed(2); }
  const n = Number(q.get("sample"));
  if (Number.isInteger(n) && n >= 1 && n <= box.children.length) box.children[n - 1].click();
}

init();
