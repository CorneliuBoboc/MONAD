import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, asdict
from typing import List

from flask import Flask, render_template, request, jsonify, send_file
from pypdf import PdfReader, PdfWriter

from google import genai
from google.genai import types

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

GEMINI_MODEL = "gemini-3.5-flash-lite"  # per spec; change if unavailable in your account


# ---------- helpers ----------

def _sanitize_filename(name: str, max_len: int = 80) -> str:
    """Make a safe filename out of arbitrary text."""
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
    for i, r in enumerate(ranges, 1):
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


# ---------- Gemini chapter detection ----------

CHAPTER_PROMPT = """You are analyzing a PDF document. Identify its logical top-level sections that would make sensible standalone PDF splits.

This includes:
- Front matter (title page, copyright, dedication, foreword, preface, acknowledgements, table of contents, etc.)
- Proper chapters (use the chapter NUMBER shown in the document, not a running index)
- Back matter (appendix, glossary, bibliography, index, colophon, etc.)

Return STRICT JSON, no prose, matching this schema:
{
  "sections": [
    {
      "title": "Human readable title as it appears in the document",
      "number": 7,              // printed chapter number if any, else null
      "kind": "frontmatter" | "chapter" | "backmatter",
      "start_page": 12,         // 1-based, inclusive
      "end_page": 34,           // 1-based, inclusive
      "filename": "07_The_Chapter_Title"   // no extension; safe chars; if number present, zero-pad to 2 digits and prefix
    }
  ]
}

Rules:
- Sections must be contiguous and cover the document in order.
- Front/back matter use number=null and filename prefixed with "00_".
- Filenames: ASCII letters/digits/underscore only, <=80 chars, no extension.
- Do not invent content not present in the PDF.
"""


def _gemini_detect_chapters(pdf_bytes: bytes, api_key: str, filename: str) -> List[dict]:
    client = genai.Client(api_key=api_key)

    # Upload PDF to Gemini Files API
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
        # Be defensive: strip code fences if any slipped through
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        data = json.loads(text)
        sections = data.get("sections") or []
        # Normalize filenames
        for s in sections:
            s["filename"] = _sanitize_filename(s.get("filename") or s.get("title") or "section")
        return sections
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


# ---------- routes ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/split/by-pages", methods=["POST"])
def split_by_pages():
    """
    Multipart form:
      files: one or more PDFs
      ranges: JSON string [{"start":1,"end":3,"name":"intro"}, ...]  (applied to each file)
    Returns a ZIP.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files uploaded"), 400

    try:
        ranges = json.loads(request.form.get("ranges", "[]"))
    except json.JSONDecodeError:
        return jsonify(error="Invalid ranges JSON"), 400
    if not ranges:
        return jsonify(error="No ranges provided"), 400

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            data = f.read()
            base = os.path.splitext(os.path.basename(f.filename or "document"))[0]
            folder = _sanitize_filename(base)
            try:
                parts = _split_by_ranges(data, ranges)
            except Exception as e:
                return jsonify(error=f"Failed on {f.filename}: {e}"), 400
            for name, blob in parts:
                zf.writestr(f"{folder}/{name}", blob)

    zip_buf.seek(0)
    return send_file(
        zip_buf, mimetype="application/zip",
        as_attachment=True, download_name="split_by_pages.zip",
    )


@app.route("/api/split/by-chapters", methods=["POST"])
def split_by_chapters():
    """
    Multipart form:
      files: one or more PDFs
      api_key: Gemini API key
    Returns a ZIP with subfolders per input file.
    """
    files = request.files.getlist("files")
    api_key = (request.form.get("api_key") or "").strip()
    if not files:
        return jsonify(error="No files uploaded"), 400
    if not api_key:
        return jsonify(error="Missing Gemini API key"), 400

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            data = f.read()
            base = os.path.splitext(os.path.basename(f.filename or "document"))[0]
            folder = _sanitize_filename(base)
            try:
                sections = _gemini_detect_chapters(data, api_key, f.filename or "document.pdf")
            except Exception as e:
                return jsonify(error=f"Gemini failed on {f.filename}: {e}"), 400

            if not sections:
                return jsonify(error=f"No sections detected in {f.filename}"), 400

            ranges = [
                {"start": s["start_page"], "end": s["end_page"], "name": s["filename"]}
                for s in sections
            ]
            try:
                parts = _split_by_ranges(data, ranges)
            except Exception as e:
                return jsonify(error=f"Split failed on {f.filename}: {e}"), 400
            for name, blob in parts:
                zf.writestr(f"{folder}/{name}", blob)

    zip_buf.seek(0)
    return send_file(
        zip_buf, mimetype="application/zip",
        as_attachment=True, download_name="split_by_chapters.zip",
    )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5009)
