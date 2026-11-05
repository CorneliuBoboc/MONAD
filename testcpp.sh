#!/usr/bin/env bash

set -e

ROOT_DIR=$PWD
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

pkill -kill $PYTHON

export WHISPER_CPP_DIR WHISPER_CPP_MODEL_DIR WHISPER_CPP_BIN

$PYTHON $APP_DIR/app.py &>$LOG_FILE &

