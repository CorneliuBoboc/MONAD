#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash lauboth.sh [--warm|--cold]" >&2
    exit 2
fi

MODE="${1:---warm}"
case "$MODE" in
    --warm|--cold) ;;
    *)
        echo "Usage: bash lauboth.sh [--warm|--cold]" >&2
        exit 2
        ;;
esac

bash "$ROOT_DIR/lau.old.both.sh" "$MODE"
bash "$ROOT_DIR/laucpp.both.sh" "$MODE"
