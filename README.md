# MONAD

  MONAD is a small local launcher + collection of self-contained Flask apps for
  media work, document conversion, and video processing. Each app lives under
  `projects/` and runs independently with its own Python requirements.

  ## Included apps

  | App | Purpose | Local port |
  | --- | --- | ---: |
  | `diarix` | Media upload, trimming, transcription, and optional diarization | 5030 |
  | `bfc` | Document conversion and split utilities with optional AI chapter/TOC generation | 5034 |
  | `pdfutils` | PDF utilities such as splitting and chapter detection | 5009 |
  | `vd` | Video download and local media processing | 5005 |

  The root `index.html` exposes quick links to the local apps. Each app may have
  its own README/spec in its project directory for detailed usage and
  limitations.

  ## Requirements

  - Linux or macOS
  - Bash and standard Unix utilities (`lsof`, `ping`)
  - Python 3.12 or 3.13
  - `pip`
  - `ffmpeg` and `ffprobe` for the media-related apps

  Some apps also depend on large packages such as Whisper, `pyannote.audio`,
  PyMuPDF, EbookLib, or `yt-dlp`. These are installed from each app's
  `requirements.txt` when `test.sh` runs.

  ## Quick start

  From the repo root, launch everything:

  ```bash
  ./lau.sh
  ```

  This script:

  - discovers apps in `projects/`
  - exports `DEMO` and `VER`
  - stops any old processes already using the configured ports
  - creates or reuses a root `.venv`
  - runs `test.sh` for each app to install dependencies and verify startup
  - waits for outbound connectivity before starting the stack

  To recreate the environment from scratch:

  ```bash
  ./lau.sh --cold
  ```

  To validate a single app manually:

  ```bash
  bash ./test.sh diarix
  bash ./test.sh bfc
  bash ./test.sh vd
  bash ./test.sh pdfutils
  ```

  Open `index.html` in a browser, or visit these URLs once launched:

  - <http://127.0.0.1:5030> (`diarix`)
  - <http://127.0.0.1:5034> (`bfc`)
  - <http://127.0.0.1:5009> (`pdfutils`)
  - <http://127.0.0.1:5005> (`vd`)

  ## Manual app startup

  Each project is self-contained. Example:

  ```bash
  cd projects/diarix
  python3.13 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python app.py
  ```

  Use the corresponding directory and requirements file for `bfc`, `pdfutils`,
  or `vd`.

  ## Environment variables

  The launcher exports:

  - `DEMO` — enabled automatically on Linux; used to select Linux-specific
    dependency behavior.
  - `VER` — Python version used by the launcher (`3.13` on Linux, `3.12`
    otherwise).

  Apps may also rely on service credentials for optional AI features, for example:

  - `ANTHROPIC_API_KEY`
  - `GEMINI_API_KEY`
  - app-specific secret keys such as `DOCUTIL_SECRET_KEY` or `MEDIA_EDITOR_SECRET`

  See each project README or source file for the exact variables supported by the
  app.

  ## Utility scripts

  - `instal` copies `zsh_aliases` into `~/.zsh_aliases` and adds the source line
    to `~/.zshrc` when needed.
  - `update.sh` removes `$HOME/MONAD`, clones the upstream repo, and enters the
    fresh copy.
  - `pacbuild.sh` builds `pac.c` and installs it as `/usr/local/bin/pac`.
  - `pac.c` timestamps each line it receives on stdin.
  - `test.sh <app>` installs that app's requirements and verifies the Flask
    process starts.

  Review scripts before running them: `lau.sh` terminates old processes on the
  configured ports, `--cold` clears venvs and pip caches, and `update.sh`
  removes an existing `$HOME/MONAD` directory.

  ## Project layout

  ```text
  .
  ├── README.md           # Project overview
  ├── index.html          # Links to local app URLs
  ├── lau.sh              # Main launcher for the full stack
  ├── test.sh             # App-specific dependency/startup check
  ├── pac.c               # Timestamping utility source
  ├── pacbuild.sh         # Build/install script for pac
  ├── instal              # Shell setup helper
  ├── update.sh           # Repo refresh helper
  ├── zsh_aliases         # Local shell aliases
  └── projects/
      ├── diarix/
      ├── bfc/
      ├── pdfutils/
      └── vd/
  ```

  ## Notes

  These apps are intended as local/personal tools rather than production
  services. They are not protected by authentication or deployment hardening by
  default; do not expose them directly to an untrusted network without adding
  those safeguards.
