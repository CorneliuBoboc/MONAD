#! /bin/bash

set -e

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APPS="morphix"
ARG="${1:---warm}"
OS="$(uname)"
# Port 5029 belongs to diarix.cpp; leave it out of this launcher's cleanup.
PORTS="5030 5034"
VER="3.12"

DEMO=false
echo "$OS"|grep -q "^Linux$" && DEMO=true
$DEMO && VER="3.14"
#$DEMO && APPS=diarix
ENGINE=mlx
$DEMO && ENGINE=faster
echo $ENGINE|grep -q "^mlx$" && VER="3.12"
$DEMO || DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib

test -n "$APPS"
test -n "$DEMO"
test -n "$OS"
test -n "$PORTS"
test -n "$VER"
case "$ARG" in
  --warm|--cold) ;;
  *)
    echo "Usage: bash lau.old.both.sh [--warm|--cold]" >&2
    exit 2
    ;;
esac

export DEMO VER

kill_old() {
  for PORT in $PORTS; do
    pids="$(lsof -i ":$PORT"|grep -v COMMAND|tr -s " " |cut -f 2 -d\ )"
    test -n "$pids" && for pid in $pids; do
      echo "killing process $pid (was using port $PORT)"
      kill -term $pid 2>/dev/null
    done
  done
}

purge_pip() {
  kill_old
  command -v deactivate  && deactivate || true
  find . -type d -iname "venv" | xargs rm -rf || true
  find . -type d -iname ".venv" | xargs rm -rf || true
  echo "Rebuilding .venv while retaining the pip cache."
  python$VER -m venv .venv
  source .venv/bin/activate
  test -n "$VIRTUAL_ENV"
  test -d "$VIRTUAL_ENV"
  export VIRTUAL_ENV
}

direct_pip() {
  kill_old
  command -v deactivate  && deactivate || true
  test -d .venv || python$VER -m venv .venv
  source .venv/bin/activate || return
  test -n "$VIRTUAL_ENV"
  test -d "$VIRTUAL_ENV"
  export VIRTUAL_ENV
}

launch_apps() {
  SCRIPT="test.old.sh"
  pip install -r projects/morphix/requirements.txt
  bash $SCRIPT $APPS
}

warm=true
[[ "$ARG" == "--cold" ]] && warm=false
echo "Please wait ..."
while ! ping -c 1 8.8.8.8 &>/dev/null; do
  sleep 40
done
if $warm; then
  if direct_pip && launch_apps; then
    true
  else
    purge_pip && launch_apps
  fi
else
  purge_pip && launch_apps
fi

echo "Morphix has been launched"
echo "See it on port 5034"
echo "Done."
