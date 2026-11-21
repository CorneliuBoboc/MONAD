#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/morphix.cpp"
PORT="${PORT:-5034}"
LOG_FILE="${TMPDIR:-/tmp}/morphix-cpp.log"

cmake -S "$ROOT_DIR/projects/morphix.cpp" -B "$BUILD_DIR" >/dev/null
cmake --build "$BUILD_DIR" -j2 >/dev/null

pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$pids" ]]; then
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done <<< "$pids"
  sleep 1
fi

PORT="$PORT" "$BUILD_DIR/morphix_cpp" >"$LOG_FILE" 2>&1 &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null || true' EXIT

for _ in {1..20}; do
  if curl --fail --silent "http://127.0.0.1:$PORT/" >/dev/null; then
    break
  fi
  sleep 0.25
done

curl --fail --silent "http://127.0.0.1:$PORT/" | grep -q "Document Utilities"
echo "Morphix C++ started and responded on port $PORT."
