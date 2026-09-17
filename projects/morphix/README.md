# Document Utilities

A single-file Flask app (`app.py`) for converting and splitting documents,
with optional AI-assisted chapter splitting and table-of-contents generation.
The entire UI (HTML/CSS/JS) is embedded in `app.py` — there are no template
files or static assets to manage.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**.

Two dependencies are optional-but-recommended:

- **PyMuPDF (`fitz`)** — without it, PDF reading falls back to `pypdf`, which
  extracts plain text only (no heading detection, weaker paragraph
  boundaries). Install PyMuPDF for meaningfully better PDF conversion/split
  quality.
- **EbookLib** — required for any EPUB support (reading, writing, splitting).
  There is no fallback for EPUB.

If a library is missing, the app still starts; only operations that need it
will return a clear error message telling you what to install.

## What it does

1. **Upload** a PDF, DOCX, AZW3, EPUB, Markdown, HTML, or TXT file (150 MB limit).
2. **Convert** supported documents to the available formats. AZW3 files use
   KindleUnpack and can be converted to EPUB (EPUB 2) only.
3. **Split by page/range** — page ranges for PDF, chapter ranges for EPUB, or
   ranges over top-level (`#` / Heading 1) sections for the other formats.
4. **Split smart by chapters (AI)** — sends a structural outline (not the
   whole document) to Claude or Gemini, asks it to propose chapter
   boundaries, and splits accordingly.
5. **Make a detailed table of contents (AI)** — sends the document's text
   (headings + paragraphs, truncated if very long) to Claude or Gemini and
   asks for a deeply nested Markdown TOC.

Every operation writes to a per-browser-session temp folder. Files older than
2 hours are swept automatically by a background thread.

## AI provider setup

For the two AI-powered operations, you can either:

- Check **"Enter my own API key"** in the UI and paste a key (used only for
  that one request — never written to disk or logged), or
- Set `ANTHROPIC_API_KEY` and/or `GEMINI_API_KEY` as environment variables on
  the machine running the server, so users don't need their own keys.

Model names drift over time — the UI has an optional "Model" override field
per provider; leave it blank to use the app's built-in default
(`claude-sonnet-5` / `gemini-3.5-flash-lite`), or set it explicitly if those
become outdated.

## Conversion fidelity — please read

This tool works at the **structure level** (headings + paragraphs), not the
pixel/layout level. It intentionally does not aim to be a byte-perfect
document converter. Concretely:

- Images, tables, footnotes, columns, fonts, and precise page layout are
  **not preserved** across conversions.
- PDF heading detection is a font-size/boldness heuristic (via PyMuPDF) —
  it's good but not perfect, and is unavailable in the `pypdf` fallback mode.
- Manual "split by range" for DOCX/HTML/MD/TXT requires existing top-level
  headings to define section boundaries; if none exist, use "Split smart by
  chapters (AI)" instead.
- Very large documents are truncated before being sent to an AI provider
  (~15,000 characters for chapter-boundary analysis, ~100,000 characters for
  TOC generation) to stay within reasonable request sizes; the UI tells you
  when truncation happened.
- AZW3 conversion downloads KindleUnpack from its upstream GitHub repository
  on first use, runs it with `--epub_version=2`, and removes its per-conversion
  temporary files afterward. AZW3 files are not available for splitting or AI
  operations until converted to EPUB.

If you need pixel-perfect conversion (e.g. preserving exact DOCX layout in a
PDF), a different tool (e.g. LibreOffice headless, Pandoc + a PDF engine) is
a better fit — this app is intentionally dependency-light and pure-Python.

## Security notes

- Uploaded filenames are sanitized (`werkzeug.secure_filename`) and
  extensions are checked against an allow-list; requests are capped at
  60 MB.
- All file/output IDs are server-generated UUIDs, validated by regex before
  any filesystem lookup — user input never touches a file path directly.
- API keys entered in the UI are used only in-memory for the single request
  that needs them; they are not persisted to session state, disk, or logs.
- The Flask `secret_key` is randomly generated at startup. If you run this
  behind multiple worker processes (e.g. Gunicorn with `--workers > 1`) or
  want sessions to survive restarts, set a fixed `DOCUTIL_SECRET_KEY`
  environment variable.
- This app has no authentication and is intended for local/single-user or
  trusted-network use. Add authentication (and consider CSRF protection) in
  front of it before exposing it on the open internet.

## Project layout

```
app.py              # Flask app + embedded HTML/CSS/JS frontend
requirements.txt
README.md
```
