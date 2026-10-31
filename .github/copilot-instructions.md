# Copilot instructions for MONAD

## Repository shape

This repo is a launcher plus a small set of independent local Flask apps under `projects/`.

- `lau.sh` is the top-level orchestration entry point.
- `test.sh` is the app-level startup validator: it installs a specific app's dependencies and launches `python app.py`.
- `index.html` links to the local app URLs.
- Each app is self-contained: `app.py`, app-local `requirements*.txt`, and usually its own README or tests.
- The codebase is intentionally local/single-user tooling, not a production deployment stack.

Today the active app directories are `projects/diarix` and `projects/morphix`.

## Build, test, and validation commands

There is no repo-wide Python test suite or lint config. Validation is done via the launcher and the app-local startup/tests.

Run the full stack from the repo root:

```bash
./lau.sh
```

Recreate the root virtual environment and reinstall dependencies:

```bash
./lau.sh --cold
```

Validate a single app by name:

```bash
bash ./test.sh diarix
bash ./test.sh morphix
```

App-local tests:

```bash
cd projects/morphix
python3 -m unittest discover -s tests -v
```

Manual app startup pattern:

```bash
cd projects/diarix
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Use the same pattern for `projects/morphix` and its corresponding `requirements.txt`.

## High-level architecture

### Root launcher model

`lau.sh` owns the cross-app lifecycle:

- discovers apps under `projects/`
- exports `DEMO` and `VER` environment variables
- kills stale processes already bound to the configured ports
- creates/reuses the root `.venv`
- resolves/install dependencies via `requirements.txt` and the `requirements.template_*` flow
- runs `test.sh` for each app to verify startup
- waits for outbound connectivity before finishing

This means startup order, dependency setup, and port cleanup are centralized in the root launcher rather than handled per app.

### App-level architecture

Each app is a standalone Flask service with embedded frontend code, not a shared backend:

- `projects/diarix/app.py` is a media editor: upload or load media, cut chunks with ffmpeg, transcribe with Whisper/faster-whisper/MLX options, and optionally diarize
- `projects/morphix/app.py` is a document converter and splitter with optional AI chapter/TOC generation
- app UIs are primarily embedded in the same `app.py`; there are no broad template/static asset conventions to follow
- app-specific requirements and any app-specific tests live next to the app code

The repo is intentionally not structured as a multi-package application or a shared library. Each app owns its own dependencies, runtime assumptions, and validation path.

## Key conventions

- Prefer app-local changes: keep modifications in the relevant `projects/<app>/` directory unless the work clearly belongs at the repo root.
- Do not assume a shared backend or a monorepo framework; each app is independent and starts as `python app.py`.
- `lau.sh` and `test.sh` are the repository's operational conventions for environment preparation and startup verification.
- Root-level dependency management is intentionally minimal; large runtime dependencies are installed per app via each app's `requirements*.txt`.
- Local tools are designed for trusted network or single-user use; they are not hardened production services by default.
- AI-powered features may depend on `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or app-specific secret keys; check the app README/source before editing those integrations.
- `morphix` includes a real unit-test path (`projects/morphix/tests`) for regression checks; use the existing test conventions there rather than inventing a new test harness.

## Repo-specific notes

- The launcher intentionally targets ports 5030 and 5034 for the currently active apps.
- `DEMO` and `VER` are set by `lau.sh`; do not hardcode or duplicate that behavior when making environment-sensitive changes.
- `ffmpeg` is part of the runtime assumptions for media-related work and is checked by `test.sh` before app startup.
- For app changes, update the relevant app's requirements if new Python packages are introduced.
