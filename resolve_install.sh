#!/usr/bin/env bash
# resolve_install.sh
# Usage: ./resolve_install.sh [requirements.txt] [venv_dir]
#
# Resolves ALL requirements in a single pip run (so the resolver sees every
# constraint at once), writes the chosen versions to requirements.lock.txt,
# installs them in an isolated virtualenv, and verifies with `pip check`.

set -euo pipefail
IFS=$'\n\t'

REQ_FILE="${1:-requirements.txt}"
VENV_DIR="${2:-.venv}"
LOCK_FILE="requirements.lock.txt"
VER=3.14

if [[ ! -f "$REQ_FILE" ]]; then
    echo "Error: '$REQ_FILE' not found." >&2
    exit 1
fi

command -v python$VER >/dev/null 2>&1 || { echo "Error: python$VER is required." >&2; exit 1; }

# Isolated environment: never touches system or user site-packages.
if [[ ! -d "$VENV_DIR" ]]; then
    python$VER -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python$VER -m pip install --quiet --upgrade pip

REPORT="$(mktemp)"
trap 'rm -f "$REPORT"' EXIT

echo ">> Resolving all requirements together..."
if ! python$VER -m pip install \
        --dry-run \
        --ignore-installed \
        --quiet \
        --report "$REPORT" \
        -r "$REQ_FILE"; then
    echo "Error: no set of versions satisfies all requirements together." >&2
    echo "Check the conflicting packages reported above." >&2
    exit 1
fi

# Extract the exact versions the resolver selected (direct + transitive).
python$VER - "$REPORT" "$LOCK_FILE" <<'PY'
import json
import sys

report_path, lock_path = sys.argv[1], sys.argv[2]
with open(report_path, encoding="utf-8") as f:
    report = json.load(f)

pins = sorted(
    {
        f"{item['metadata']['name']}=={item['metadata']['version']}"
        for item in report.get("install", [])
    },
    key=str.lower,
)

with open(lock_path, "w", encoding="utf-8") as f:
    f.write("\n".join(pins) + "\n")

print(f">> Resolved {len(pins)} packages -> {lock_path}")
PY

echo ">> Installing resolved versions..."
python$VER -m pip install -r "$LOCK_FILE"

echo ">> Verifying dependency consistency..."
python$VER -m pip check

echo ">> Done. Activate with: source $VENV_DIR/bin/activate"
