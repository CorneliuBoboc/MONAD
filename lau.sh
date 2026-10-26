#! /bin/bash

set -e

APPS="$(ls projects)"
ARG="$1"
OS="$(uname)"
PORTS="5030 5034"
VER="3.12"

DEMO=false
echo "$OS"|grep -q "^Linux$" && DEMO=true
$DEMO && VER="3.14"
$DEMO && APPS=diarix

test -n "$APPS"
test -n "$DEMO"
test -n "$OS"
test -n "$PORTS"
test -n "$VER"

export DEMO VER

kill_old() {
  for PORT in $PORTS; do
    pids="$(lsof -ti ":$PORT" 2>/dev/null || true)"
    if test -n "$pids"; then
      echo "Stopping existing listener on port $PORT."
      for pid in $pids; do
        kill -term "$pid" 2>/dev/null || true
      done
    fi
  done
}

purge_pip() {
  kill_old
  command -v deactivate >/dev/null 2>&1 && deactivate >/dev/null 2>&1 || true
  find . -type d -iname "venv" | xargs rm -rf >/dev/null 2>&1 || true
  find . -type d -iname ".venv" | xargs rm -rf >/dev/null 2>&1 || true
  rm -rf "$HOME/.cache/pip" >/dev/null 2>&1 || true
  if ! python$VER -m venv .venv >/dev/null 2>&1; then
    echo "Could not create the virtual environment." >&2
    return 1
  fi
  if ! source .venv/bin/activate 2>/dev/null; then
    echo "Could not activate the virtual environment." >&2
    return 1
  fi
  test -n "$VIRTUAL_ENV"
  test -d "$VIRTUAL_ENV"
  export VIRTUAL_ENV
  pip install --upgrade pip >/dev/null 2>&1 || true
}

direct_pip() {
  kill_old
  command -v deactivate >/dev/null 2>&1 && deactivate >/dev/null 2>&1 || true
  if ! test -d .venv && ! python$VER -m venv .venv >/dev/null 2>&1; then
    echo "Could not create the virtual environment." >&2
    return 1
  fi
  if ! source .venv/bin/activate 2>/dev/null; then
    echo "Could not activate the virtual environment." >&2
    return 1
  fi
  test -n "$VIRTUAL_ENV"
  test -d "$VIRTUAL_ENV"
  export VIRTUAL_ENV
}

launch_apps() {
  SCRIPT="test.sh"
  for APP in $APPS; do
    test -n "$APP"
    bash "$SCRIPT" "$APP" && echo "Launched $APP ok"
  done
}

warm=true
test -n "$ARG" && echo "$ARG"|grep -q "^--cold$" && warm=false
echo "MONAD launcher started."
echo "Platform: $OS ($(uname -m)); Python: $VER"
echo "Cold install: $([[ "$warm" == false ]] && echo yes || echo no)"
echo "Waiting for network connectivity."
while ! ping -c 1 8.8.8.8 >/dev/null 2>&1; do
  sleep 40
done
if $warm; then
  if direct_pip && launch_apps; then
    true
  else
    if ! purge_pip || ! launch_apps; then
      echo "MONAD launch failed; inspect the safe diagnostics above." >&2
      exit 1
    fi
  fi
else
  if ! purge_pip || ! launch_apps; then
    echo "MONAD launch failed; inspect the safe diagnostics above." >&2
    exit 1
  fi
fi

echo "All configured apps passed process startup checks."
echo "MONAD launcher finished."
