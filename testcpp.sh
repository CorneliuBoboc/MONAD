#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/projects/diarix.cpp"
PYTHON="$ROOT_DIR/.venv-cpp/bin/python"
WHISPER_CPP_DIR="${WHISPER_CPP_DIR:-$HOME/code/Danielzapirtan/whisper.cpp}"
WHISPER_CPP_MODEL_DIR="${WHISPER_CPP_MODEL_DIR:-$HOME/models}"
WHISPER_CPP_BIN="${WHISPER_CPP_BIN:-$WHISPER_CPP_DIR/build/bin/whisper-cli}"
MODEL_PATH="$WHISPER_CPP_MODEL_DIR/ggml-large-v3.bin"
PORT=5029
URL="http://127.0.0.1:$PORT/"
LOG_FILE="${TMPDIR:-/tmp}/diarix-cpp.log"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: $PYTHON not found. Run ./laucpp.sh first." >&2
    exit 1
fi
if [[ ! -x "$WHISPER_CPP_BIN" ]]; then
    echo "Error: whisper.cpp CLI not found: $WHISPER_CPP_BIN. Run ./laucpp.sh first." >&2
    exit 1
fi
if [[ ! -s "$MODEL_PATH" ]]; then
    echo "Error: large-v3 model not found: $MODEL_PATH. Run ./laucpp.sh first." >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is required. Install it with 'sudo apt install ffmpeg' on Ubuntu or 'brew install ffmpeg' on macOS." >&2
    exit 1
fi

export WHISPER_CPP_DIR WHISPER_CPP_MODEL_DIR WHISPER_CPP_BIN

check_app() {
    "$PYTHON" - "$URL" <<'PY'
import sys
from urllib.error import URLError
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=2) as response:
        page = response.read(200_000).decode("utf-8", errors="replace")
except (OSError, URLError):
    sys.exit(1)
if "Media editor" in page and "whisper.cpp" in page:
    sys.exit(0)
sys.exit(2)
PY
}

if check_app; then
    echo "diarix.cpp is already responding at $URL"
    exit 0
else
    check_status=$?
    if [[ "$check_status" -eq 2 ]]; then
        echo "Error: port $PORT is occupied by a different service." >&2
        exit 1
    fi
fi

cd "$APP_DIR"
nohup "$PYTHON" -c \
    'import app; app.app.run(host="0.0.0.0", port=5030, threaded=True, use_reloader=False)' \
    >"$LOG_FILE" 2>&1 </dev/null &
APP_PID=$!

for ((attempt = 1; attempt <= 60; attempt++)); do
    if check_app; then
        echo "diarix.cpp is running at $URL (pid $APP_PID)"
        echo "Server log: $LOG_FILE"
        exit 0
    else
        check_status=$?
        if [[ "$check_status" -eq 2 ]]; then
            echo "Error: port $PORT is occupied by a different service." >&2
            kill -TERM "$APP_PID" 2>/dev/null || true
            exit 1
        fi
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        echo "Error: diarix.cpp exited during startup. Last log lines:" >&2
        tail -40 "$LOG_FILE" >&2 || true
        exit 1
    fi
    sleep 1
done

echo "Error: diarix.cpp did not become ready within 60 seconds. Last log lines:" >&2
tail -40 "$LOG_FILE" >&2 || true
kill -TERM "$APP_PID" 2>/dev/null || true
exit 1
