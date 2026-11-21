#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash lau.native.sh [--warm|--cold]" >&2
    exit 2
fi

MODE="${1:---warm}"
case "$MODE" in
    --warm|--cold) ;;
    *)
        echo "Usage: bash lau.native.sh [--warm|--cold]" >&2
        exit 2
        ;;
esac

OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
    DEMO=false
    VER="3.12"
else
    DEMO=true
    VER="3.14"
fi
export DEMO VER

DIARIX_PORT=5030
MORPHIX_PORT=5034
LOG_DIR="${TMPDIR:-/tmp}"
DIARIX_LOG="$LOG_DIR/diarix-cpp.log"
MORPHIX_LOG="$LOG_DIR/morphix-cpp.log"

stop_port_listener() {
    local port="$1"
    local label="$2"
    local pids=""

    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
    fi

    if [[ -z "$pids" ]]; then
        return
    fi

    echo "Stopping prior $label listener on port $port"
    while IFS= read -r pid; do
        [[ -n "$pid" ]] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done < <(printf '%s\n' "$pids" | tr ' ' '\n' | sort -u)

    for _ in {1..20}; do
        local remaining=""
        if command -v lsof >/dev/null 2>&1; then
            remaining="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        elif command -v fuser >/dev/null 2>&1; then
            remaining="$(fuser -n tcp "$port" 2>/dev/null || true)"
        fi
        [[ -z "$remaining" ]] && return
        sleep 0.25
    done

    echo "Error: port $port is still in use." >&2
    exit 1
}

wait_for_url() {
    local name="$1"
    local url="$2"
    local attempts="${3:-120}"

    for _ in $(seq 1 "$attempts"); do
        if curl --fail --silent --show-error "$url" >/dev/null 2>&1; then
            echo "$name is responding at $url"
            return
        fi
        sleep 0.25
    done

    echo "Error: $name did not respond at $url" >&2
    return 1
}

for command_name in cmake curl; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command '$command_name' is not installed." >&2
        exit 1
    fi
done

if [[ "$MODE" == "--cold" ]]; then
    rm -rf "$ROOT_DIR/build/morphix.cpp"
fi

stop_port_listener "$DIARIX_PORT" "diarix.cpp"
stop_port_listener "$MORPHIX_PORT" "morphix.cpp"

echo "Starting native apps (DEMO=$DEMO, OS=$OS)..."
bash "$ROOT_DIR/laucpp.both.sh" "$MODE"

echo "Building morphix.cpp..."
cmake -S "$ROOT_DIR/projects/morphix.cpp" -B "$ROOT_DIR/build/morphix.cpp" >/dev/null
cmake --build "$ROOT_DIR/build/morphix.cpp" -j2 >/dev/null

echo "Starting morphix.cpp on port $MORPHIX_PORT..."
PORT="$MORPHIX_PORT" nohup "$ROOT_DIR/build/morphix.cpp/morphix_cpp" \
    >"$MORPHIX_LOG" 2>&1 &
MORPHIX_PID=$!

wait_for_url "diarix.cpp" "http://127.0.0.1:$DIARIX_PORT/" 240 || {
    echo "See diarix.cpp log: $DIARIX_LOG" >&2
    exit 1
}
wait_for_url "morphix.cpp" "http://127.0.0.1:$MORPHIX_PORT/" 120 || {
    echo "See morphix.cpp log: $MORPHIX_LOG" >&2
    kill -TERM "$MORPHIX_PID" 2>/dev/null || true
    exit 1
}

echo "diarix.cpp: http://127.0.0.1:$DIARIX_PORT/"
echo "morphix.cpp: http://127.0.0.1:$MORPHIX_PORT/"
echo "Logs: $DIARIX_LOG and $MORPHIX_LOG"
