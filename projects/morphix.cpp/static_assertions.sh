#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT_DIR/app.cpp"

grep -q 'poppler::document::load_from_file' "$APP"
grep -q 'zip_open' "$APP"
grep -q 'render_pdf' "$APP"
grep -q 'render_docx' "$APP"
grep -q 'render_epub' "$APP"
grep -q 'x-api-key:' "$APP"
grep -q 'anthropic-version:' "$APP"
grep -q 'x-goog-api-key:' "$APP"
grep -q '/api/ai/split/start' "$APP"
grep -q '/api/jobs/' "$APP"
grep -q '/api/download/source/' "$APP"
grep -q '/api/download-all' "$APP"
grep -q 'qpdf --empty --pages' "$APP"
grep -q '/api/ai/split/start' "$APP"
grep -q 'run_ai_split_job' "$APP"
grep -q 'call_gemini' "$APP"
grep -q 'x-goog-api-key:' "$APP"
grep -q 'parse_sections' "$APP"
grep -q '/api/jobs/' "$APP"

if grep -q 'Bearer + key' "$APP"; then
  echo "Found invalid bearer-only AI authentication." >&2
  exit 1
fi

echo "Static native implementation assertions passed."
