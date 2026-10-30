#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_CPP_URL="https://github.com/Danielzapirtan/whisper.cpp.git"
WHISPER_CPP_BRANCH="Danielzapirtan"
WHISPER_CPP_DIR="${WHISPER_CPP_DIR:-$HOME/code/Danielzapirtan/whisper.cpp}"
WHISPER_CPP_MODEL_DIR="${WHISPER_CPP_MODEL_DIR:-$HOME/models}"
WHISPER_CPP_BIN="${WHISPER_CPP_BIN:-$WHISPER_CPP_DIR/build/bin/whisper-cli}"
VENV_DIR="$ROOT_DIR/.venv-cpp"

if [[ ! -d "$WHISPER_CPP_DIR" ]]; then
    mkdir -p "$(dirname "$WHISPER_CPP_DIR")"
    echo "Cloning whisper.cpp branch '$WHISPER_CPP_BRANCH' into $WHISPER_CPP_DIR"
    git clone --branch "$WHISPER_CPP_BRANCH" --single-branch \
        "$WHISPER_CPP_URL" "$WHISPER_CPP_DIR"
elif [[ ! -d "$WHISPER_CPP_DIR/.git" ]]; then
    echo "Error: $WHISPER_CPP_DIR exists but is not a git checkout." >&2
    exit 1
fi

origin_url="$(git -C "$WHISPER_CPP_DIR" remote get-url origin)"
case "$origin_url" in
    https://github.com/Danielzapirtan/whisper.cpp|\
    https://github.com/Danielzapirtan/whisper.cpp.git|\
    git@github.com:Danielzapirtan/whisper.cpp|\
    git@github.com:Danielzapirtan/whisper.cpp.git)
        ;;
    *)
        echo "Error: unexpected whisper.cpp origin: $origin_url" >&2
        exit 1
        ;;
esac

current_branch="$(git -C "$WHISPER_CPP_DIR" branch --show-current)"
if [[ "$current_branch" != "$WHISPER_CPP_BRANCH" ]]; then
    echo "Error: $WHISPER_CPP_DIR is on branch '$current_branch'; expected '$WHISPER_CPP_BRANCH'." >&2
    echo "Switch branches manually; laucpp.sh will not change an existing checkout." >&2
    exit 1
fi

for command_name in git cmake; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command '$command_name' is not installed." >&2
        exit 1
    fi
done

case "$(uname -s)" in
    Darwin)
        metal=ON
        ;;
    Linux)
        metal=OFF
        ;;
    *)
        echo "Error: laucpp.sh supports Linux and macOS." >&2
        exit 1
        ;;
esac

if [[ ! -x "$WHISPER_CPP_BIN" ]]; then
    if [[ "$WHISPER_CPP_BIN" != "$WHISPER_CPP_DIR/build/bin/whisper-cli" ]]; then
        echo "Error: configured WHISPER_CPP_BIN is not executable: $WHISPER_CPP_BIN" >&2
        exit 1
    fi
    echo "Building whisper.cpp CLI for this machine..."
    cmake -S "$WHISPER_CPP_DIR" -B "$WHISPER_CPP_DIR/build" \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_SERVER=OFF \
        -DGGML_METAL="$metal"
    cmake --build "$WHISPER_CPP_DIR/build" --config Release \
        --target whisper-cli --parallel
fi

mkdir -p "$WHISPER_CPP_MODEL_DIR"
model_path="$WHISPER_CPP_MODEL_DIR/ggml-large-v3.bin"
if [[ -f "$model_path" ]]; then
    if [[ ! -s "$model_path" ]]; then
        echo "Error: model file exists but is empty: $model_path" >&2
        exit 1
    fi
    echo "Found existing large-v3 model at $model_path"
else
    echo "Downloading the large-v3 GGML model into $WHISPER_CPP_MODEL_DIR"
    sh "$WHISPER_CPP_DIR/models/download-ggml-model.sh" large-v3 \
        "$WHISPER_CPP_MODEL_DIR"
fi

if [[ ! -s "$model_path" ]]; then
    echo "Error: large-v3 model download did not produce $model_path" >&2
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
elif command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.13)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Error: Python 3.12 or newer is required." >&2
    exit 1
fi

"$PYTHON_BIN" -c 'import sys; sys.version_info >= (3, 12) or sys.exit("Python 3.12 or newer is required.")'
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/projects/diarix.cpp/requirements.txt"
"$VENV_DIR/bin/python" -m pip check

export WHISPER_CPP_DIR WHISPER_CPP_MODEL_DIR WHISPER_CPP_BIN
echo "Starting diarix.cpp..."
bash "$ROOT_DIR/testcpp.sh"
