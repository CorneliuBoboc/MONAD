#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${MONAD_DIR:-$HOME/MONAD}"
REPO_URL="https://github.com/CorneliuBoboc/MONAD.git"

if [[ ! -e "$REPO_DIR" ]]; then
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
    echo "Cloned MONAD into $REPO_DIR"
    exit 0
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "Error: $REPO_DIR exists but is not a git checkout; refusing to replace it." >&2
    exit 1
fi

origin_url="$(git -C "$REPO_DIR" remote get-url origin)"
case "$origin_url" in
    https://github.com/CorneliuBoboc/MONAD|\
    https://github.com/CorneliuBoboc/MONAD.git|\
    git@github.com:CorneliuBoboc/MONAD|\
    git@github.com:CorneliuBoboc/MONAD.git)
        ;;
    *)
        echo "Error: $REPO_DIR origin is '$origin_url', not $REPO_URL." >&2
        exit 1
        ;;
esac

if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    echo "Error: $REPO_DIR has local changes; commit or stash them before updating." >&2
    exit 1
fi

git -C "$REPO_DIR" pull --ff-only
