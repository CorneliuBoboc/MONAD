import io
import json
import os
import re
import zipfile
from typing import List

from flask import Flask, render_template_string, request, jsonify, send_file
from pypdf import PdfReader, PdfWriter

from google import genai
from google.genai import types

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

GEMINI_MODEL = "gemini-3.5-flash-lite"  # per spec; change if unavailable in your account


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str, max_len: int = 80) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._-") or "untitled"
    return name[:max_len]


def _read_pdf(data: bytes) -> PdfReader:
    return PdfReader(io.BytesIO(data))


def _split_by_ranges(data: bytes, ranges: List[dict]) -> List[tuple]:
    """
    ranges: [{"start": 1, "end": 3, "name": "intro"}, ...] 1-based inclusive.
    Returns [(filename, bytes), ...]
    """
    reader = _read_pdf(data)
    n = len(reader.pages)
    out = []
    for r in ranges:
        s = max(1, int(r["start"]))
        e = min(n, int(r["end"]))
        if s > e:
            continue
        writer = PdfWriter()
        for p in range(s - 1, e):
            writer.add_page(reader.pages[p])
        base = _sanitize_filename(r.get("name") or f"pages_{s}-{e}")
        buf = io.BytesIO()
        writer.write(buf)
        out.append((f"{base}.pdf", buf.getvalue()))
    return out


# ---------------------------------------------------------------------------
# Gemini chapter detection
# ---------------------------------------------------------------------------

CHAPTER_PROMPT = """You are analyzing a PDF document. Identify every top-level section that should become its own standalone PDF.

CRITICAL: Do NOT group all front matter into one section, and do NOT group all back matter into one. Each distinct front-matter or back-matter item must be its OWN section with its OWN filename. Examples of separate front-matter sections: cover, copyright page, dedication, table of contents, foreword, preface, acknowledgements. Examples of separate back-matter sections: appendix A, appendix B, glossary, bibliography, index, colophon, about the author.

For proper chapters, use the chapter NUMBER shown in the document (not a running index).

Return STRICT JSON, no prose, matching this schema:
{
  "sections": [
    {
      "title": "Human readable title as it appears in the document",
      "number": 7,              // printed chapter number if any, else null
      "kind": "frontmatter" | "chapter" | "backmatter",
      "start_page": 12,         // 1-based, inclusive
      "end_page": 34,           // 1-based, inclusive
      "filename": "07_The_Chapter_Title"   // no extension; safe chars; zero-pad number to 2 digits when present
    }
  ]
}

Rules:
- Sections must be contiguous and cover the document in order.
- Every distinct front/back-matter item is its own section (e.g. "00_cover", "00_copyright", "00_toc", "00_foreword", "99_glossary", "99_index").
- Front/back matter use number=null and filename prefixed with "00_" (front) or "99_" (back).
- Filenames: ASCII letters/digits/underscore only, <=80 chars, no extension.
- Do not invent content not present in the PDF.
"""


def _gemini_detect_chapters(pdf_bytes: bytes, api_key: str, filename: str) -> List[dict]:
    client = genai.Client(api_key=api_key)

    uploaded = client.files.upload(
        file=io.BytesIO(pdf_bytes),
        config=types.UploadFileConfig(mime_type="application/pdf", display_name=filename),
    )

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_uri(file_uri=uploaded.uri, mime_type="application/pdf"),
                CHAPTER_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        text = resp.text or ""
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        data = json.loads(text)
        sections = data.get("sections") or []
        for s in sections:
            s["filename"] = _sanitize_filename(s.get("filename") or s.get("title") or "section")
        return sections
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Embedded frontend
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PDF Utilities</title>
<style>
:root {
  --fg: #1c1c1e;
  --muted: #6b6b70;
  --border: #d9d9de;
  --accent: #2563eb;
  --bg: #fafafa;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--fg);
  background: var(--bg);
}
header {
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  background: #fff;
}
header h2 { margin: 0; font-weight: 600; }
.main-container {
  max-width: 820px;
  margin: 24px auto;
  padding: 0 16px;
}
.tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--border);
}
.tab {
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  padding: 8px 14px;
  cursor: pointer;
  font: inherit;
  color: var(--muted);
  border-radius: 6px 6px 0 0;
}
.tab.active {
  color: var(--fg);
  background: #fff;
  border-color: var(--border);
  border-bottom: 1px solid #fff;
  margin-bottom: -1px;
}
.panel {
  display: none;
  background: #fff;
  border: 1px solid var(--border);
  border-top: none;
  padding: 20px;
  border-radius: 0 0 6px 6px;
}
.panel.active { display: block; }
input[type="password"], input[type="file"], textarea {
  width: 100%;
  padding: 8px 10px;
  font: inherit;
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-top: 4px;
}
textarea { font-family: ui-monospace, Menlo, monospace; }
button {
  font: inherit;
  padding: 8px 14px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
}
button.primary {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
  margin-top: 12px;
}
button.primary:disabled { opacity: .5; cursor: not-allowed; }
.row { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
.status { color: var(--muted); font-size: .9em; }
.hint { color: var(--muted); font-size: .9em; }
.output {
  margin-top: 16px;
  padding: 12px;
  border-radius: 6px;
  background: #fff;
  border: 1px solid var(--border);
  display: none;
  white-space: pre-wrap;
  font-family: ui-monospace, Menlo, monospace;
  font-size: .9em;
}
.output.show { display: block; }
.output.error { border-color: #e11d48; color: #b91c1c; }
#file-info { margin-top: 12px; color: var(--muted); }
</style>
</head>
<body>
  <header><h2>PDF Utilities</h2></header>
  <main class="main-container">
    <nav class="tabs" role="tablist">
      <button class="tab active" data-tab="load"     role="tab">Load document</button>
      <button class="tab"        data-tab="split-ch" role="tab">Smart split by chapters</button>
      <button class="tab"        data-tab="split-pg" role="tab">Split by page(range)s</button>
    </nav>

    <section id="tab-load" class="panel active">
      <p>Select a single PDF file. It will be used by whichever split tab you choose next.</p>
      <input type="file" id="file" accept="application/pdf">
      <div id="file-info"></div>
    </section>

    <section id="tab-split-ch" class="panel">
      <h3>Smart split by chapters (Gemini)</h3>
      <label>
        Gemini API Key
        <input type="password" id="api-key" placeholder="paste your API key" autocomplete="off">
      </label>
      <div class="row">
        <button id="save-key">Save key</button>
        <button id="remove-key">Remove stored key</button>
        <span id="key-status" class="status"></span>
      </div>
      <p class="hint">The key is stored only in your browser's localStorage.</p>
      <button id="do-chapters" class="primary">Split by chapters &rarr; download ZIP</button>
    </section>

    <section id="tab-split-pg" class="panel">
      <h3>Split by page(range)s</h3>
      <p>Enter one range per line, format <code>start-end</code> or <code>start-end:name</code>.
         Example: <code>1-3:cover</code></p>
      <textarea id="ranges" rows="8" placeholder="1-1:cover
2-2:copyright
3-10:preface"></textarea>
      <button id="do-pages" class="primary">Split by pages &rarr; download ZIP</button>
    </section>

    <div id="output" class="output"></div>
  </main>

<script>
const KEY_STORAGE = "pdfutils.geminiApiKey";

// ---- tabs ----
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach(p => {
      p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab);
    });
  });
});

// ---- file input ----
const fileInput = document.getElementById("file");
const fileInfo  = document.getElementById("file-info");
fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  fileInfo.textContent = f ? `${f.name} \u2014 ${(f.size / 1024).toFixed(0)} KB` : "";
});

// ---- output helper ----
const output = document.getElementById("output");
function showOutput(msg, isError = false) {
  output.textContent = msg;
  output.classList.add("show");
  output.classList.toggle("error", isError);
}

// ---- API key handling ----
const keyInput  = document.getElementById("api-key");
const keyStatus = document.getElementById("key-status");
const savedKey = localStorage.getItem(KEY_STORAGE);
if (savedKey) {
  keyInput.value = savedKey;
  keyStatus.textContent = "key loaded from localStorage";
}
document.getElementById("save-key").addEventListener("click", () => {
  const k = keyInput.value.trim();
  if (!k) { keyStatus.textContent = "nothing to save"; return; }
  localStorage.setItem(KEY_STORAGE, k);
  keyStatus.textContent = "key saved";
});
document.getElementById("remove-key").addEventListener("click", () => {
  localStorage.removeItem(KEY_STORAGE);
  keyInput.value = "";
  keyStatus.textContent = "stored key removed";
});

// ---- download helper ----
async function postAndDownload(url, formData, button, label) {
  const f = fileInput.files[0];
  if (!f) {
    showOutput("Load a PDF first (tab: Load document).", true);
    return;
  }
  formData.append("file", f, f.name);

  const prev = button.textContent;
  button.disabled = true;
  button.textContent = "Working\u2026";
  try {
    const resp = await fetch(url, { method: "POST", body: formData });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try { msg = (await resp.json()).error || msg; } catch (_) {}
      showOutput("Error: " + msg, true);
      return;
    }
    const blob = await resp.blob();
    const cd = resp.headers.get("Content-Disposition") || "";
    const m = /filename="?([^"]+)"?/.exec(cd);
    const name = m ? m[1] : label;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    showOutput(`Done. Downloaded ${name}.`);
  } catch (e) {
    showOutput("Network error: " + e.message, true);
  } finally {
    button.disabled = false;
    button.textContent = prev;
  }
}

// ---- split by pages ----
function parseRanges(text) {
  const out = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const m = /^(\d+)\s*-\s*(\d+)\s*(?::\s*(.+))?$/.exec(line);
    if (!m) throw new Error(`Bad range line: "${line}"`);
    out.push({
      start: parseInt(m[1], 10),
      end:   parseInt(m[2], 10),
      name:  (m[3] || "").trim() || `pages_${m[1]}-${m[2]}`,
    });
  }
  return out;
}

document.getElementById("do-pages").addEventListener("click", async (ev) => {
  let ranges;
  try { ranges = parseRanges(document.getElementById("ranges").value); }
  catch (e) { showOutput(e.message, true); return; }
  if (!ranges.length) { showOutput("Enter at least one page range.", true); return; }

  const fd = new FormData();
  fd.append("ranges", JSON.stringify(ranges));
  await postAndDownload("/api/split/by-pages", fd, ev.currentTarget, "split_by_pages.zip");
});

// ---- split by chapters ----
document.getElementById("do-chapters").addEventListener("click", async (ev) => {
  const key = (keyInput.value || localStorage.getItem(KEY_STORAGE) || "").trim();
  if (!key) { showOutput("Enter a Gemini API key first.", true); return; }
  const fd = new FormData();
  fd.append("api_key", key);
  await postAndDownload("/api/split/by-chapters", fd, ev.currentTarget, "split_by_chapters.zip");
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/split/by-pages", methods=["POST"])
def split_by_pages():
    """
    Multipart form:
      file:   single PDF
      ranges: JSON string [{"start":1,"end":3,"name":"intro"}, ...]
    Returns a ZIP.
    """
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file uploaded"), 400

    try:
        ranges = json.loads(request.form.get("ranges", "[]"))
    except json.JSONDecodeError:
        return jsonify(error="Invalid ranges JSON"), 400
    if not ranges:
        return jsonify(error="No ranges provided"), 400

    data = f.read()
    try:
        parts = _split_by_ranges(data, ranges)
    except Exception as e:
        return jsonify(error=f"Split failed: {e}"), 400

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in parts:
            zf.writestr(name, blob)
    zip_buf.seek(0)

    base = os.path.splitext(os.path.basename(f.filename or "document"))[0]
    download_name = f"{_sanitize_filename(base)}_split_by_pages.zip"
    return send_file(
        zip_buf, mimetype="application/zip",
        as_attachment=True, download_name=download_name,
    )


@app.route("/api/split/by-chapters", methods=["POST"])
def split_by_chapters():
    """
    Multipart form:
      file:    single PDF
      api_key: Gemini API key
    Returns a ZIP.
    """
    f = request.files.get("file")
    api_key = (request.form.get("api_key") or "").strip()
    if not f:
        return jsonify(error="No file uploaded"), 400
    if not api_key:
        return jsonify(error="Missing Gemini API key"), 400

    data = f.read()
    try:
        sections = _gemini_detect_chapters(data, api_key, f.filename or "document.pdf")
    except Exception as e:
        return jsonify(error=f"Gemini failed: {e}"), 400

    if not sections:
        return jsonify(error="No sections detected"), 400

    ranges = [
        {"start": s["start_page"], "end": s["end_page"], "name": s["filename"]}
        for s in sections
    ]
    try:
        parts = _split_by_ranges(data, ranges)
    except Exception as e:
        return jsonify(error=f"Split failed: {e}"), 400

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in parts:
            zf.writestr(name, blob)
    zip_buf.seek(0)

    base = os.path.splitext(os.path.basename(f.filename or "document"))[0]
    download_name = f"{_sanitize_filename(base)}_split_by_chapters.zip"
    return send_file(
        zip_buf, mimetype="application/zip",
        as_attachment=True, download_name=download_name,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5009, debug=True)
