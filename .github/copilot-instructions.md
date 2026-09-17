# Copilot instructions for MONAD

## Repository shape

This repo is a launcher + collection of independent local Flask apps under `projects/`.

- `lau.sh` is the main orchestration entry point.
- `test.sh` validates a single app by installing that app's requirements and starting `python app.py`.
- `index.html` is a small landing page pointing at the local app URLs.
- Each app is self-contained: usually a `requirements.txt` and a single `app.py` (often with embedded HTML/CSS/JS, not templates/static folders).
- The project is intentionally local/single-user tooling, not a production web deployment.

## Build, test, and validation commands

There is no formal Python test suite or lint configuration in this repo. The practical validation paths are the launcher and per-app startup checks documented by the project.

Run the full stack from the repo root:

```bash
./lau.sh
```

Recreate the virtual environment and reinstall dependencies:

```bash
./lau.sh --cold
```

Validate one app by name:

```bash
bash ./test.sh diarix
bash ./test.sh bfc
bash ./test.sh vd
```

For a manual app startup, follow the pattern used by each project:

```bash
cd projects/diarix
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The same pattern applies to `bfc` and `vd` using their respective directories and requirements files.

## High-level architecture

### Root launcher model

`lau.sh` does the coordination work that matters across the whole repo:

- discovers apps under `projects/`
- exports `DEMO` and `VER` environment variables
- kills any old processes listening on the app ports
- creates/reuses a root `.venv`
- runs `test.sh` for each app to install dependencies and verify startup
- waits for network connectivity before launching

This means app startup order, environment setup, and port cleanup are managed centrally rather than by each app individually.

### App-level architecture

Each app under `projects/` is functionally independent and usually follows the same pattern:

- a Flask app in `app.py`
- embedded UI code (HTML/CSS/JS) in the same file
- a project-specific `requirements.txt`
- optional documentation such as a `README.md` or `spec.md`

The repo is not organized around a shared backend or library. Instead, each application is a standalone tool with its own dependency set and operational assumptions.

Examples from the repo:

- `projects/diarix`: media editor / cut + transcribe workflow
- `projects/bfc`: document utility app with optional AI chapter and TOC generation
- `projects/vd`: video download / media processing app
- `projects/pdfutils`: document conversion utility app

The root `index.html` simply exposes links to the local app URLs and is not the application logic itself.

## Key conventions

- All main apps are Flask apps and are expected to run directly with `python app.py`.
- UI code is embedded in `app.py` rather than split into templates/static folders; keep that pattern unless the project explicitly documents a different structure.
- App-specific dependencies are isolated in each project's `requirements.txt` rather than a repo-wide package manifest.
- `lau.sh` and `test.sh` are the repo's operational conventions for environment setup, startup verification, and port cleanup.
- The apps are local tools intended for trusted-network or single-user use. Do not treat them as production-ready without adding auth, rate limits, and deployment safeguards.
- AI-enabled features may rely on environment variables like `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or app-specific secret keys; check the project README/spec before adding or changing those integrations.

## Notes for future sessions

When modifying code, prefer the same app-local structure used by the project: keep the work within the relevant `projects/<app>/` directory, update the app's requirements if a new library is added, and validate with the project startup pattern rather than introducing an unrelated monorepo framework.
