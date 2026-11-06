#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_CPP_URL="https://github.com/Danielzapirtan/whisper.cpp.git"
WHISPER_CPP_BRANCH="Danielzapirtan"
WHISPER_CPP_DIR="${WHISPER_CPP_DIR:-$HOME/code/Danielzapirtan/whisper.cpp}"
WHISPER_CPP_MODEL_DIR="${WHISPER_CPP_MODEL_DIR:-$HOME/models}"
WHISPER_CPP_BIN="${WHISPER_CPP_BIN:-$WHISPER_CPP_DIR/build/bin/whisper-cli}"
VENV_DIR="$ROOT_DIR/.venv-cpp"
PORT=5029
MODE="${1:---warm}"

cd "$ROOT_DIR"

case "$MODE" in
    --warm|--cold) ;;
    *)
        echo "Usage: bash laucpp.both.sh [--warm|--cold]" >&2
        exit 2
        ;;
esac

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
    echo "Switch branches manually; laucpp.both.sh will not change an existing checkout." >&2
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
        echo "Error: laucpp.both.sh supports Linux and macOS." >&2
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

for command_name in lsof; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command '$command_name' is not installed." >&2
        exit 1
    fi
done

cd "$ROOT_DIR"
if [[ "$MODE" == "--cold" ]]; then
    rm -rf "$VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"
reqfile="projects/diarix.cpp/requirements.txt"
nreq=$(cat "$reqfile" | wc -l)
if [[ "$(uname -s)" == Linux ]] \
    && ! python -c "import torch, torchaudio" >/dev/null 2>&1; then
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
nf=1
for n in $(seq "$nf" "$nreq"); do
    cat "$reqfile" | head -n "$n" >/tmp/tmp
    r=$(tail -n 1 /tmp/tmp)
    pip show "$r" || pip install "$r"
done
pip check

stop_port_listener() {
    local pids
    pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        return
    fi

    while IFS= read -r pid; do
        [[ -n "$pid" ]] || continue
        echo "Stopping prior diarix.cpp listener $pid on port $PORT"
        kill -TERM "$pid" 2>/dev/null || true
    done <<< "$pids"

    for _ in {1..20}; do
        if ! lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
            return
        fi
        sleep 1
    done
    echo "Error: port $PORT is still in use; not starting diarix.cpp." >&2
    exit 1
}

export WHISPER_CPP_DIR WHISPER_CPP_MODEL_DIR WHISPER_CPP_BIN
stop_port_listener
echo "Starting diarix.cpp..."
bash "$ROOT_DIR/testcpp.sh"
