#! /bin/bash

set -e

APPS="$(ls projects)"
ARG="$1"
OS="$(uname)"
PORTS="5030 5034 5009 5005"
VER="3.12"

DEMO=false
echo "$OS"|grep -q "^Linux$" && DEMO=true
$DEMO && VER="3.13"
#$DEMO && APPS=diarix

test -n "$APPS"
test -n "$DEMO"
test -n "$OS"
test -n "$PORTS"
test -n "$VER"

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
  rm -rf $HOME/.cache/pip || true
  python$VER -m venv .venv
  source .venv/bin/activate
  test -n "$VIRTUAL_ENV"
  test -d "$VIRTUAL_ENV"
  export VIRTUAL_ENV
  pip install --upgrade pip  || true
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
  SCRIPT="test.sh"
  for APP in $APPS; do
    test -n "$APP"
    bash "$SCRIPT" "$APP" && echo "Launched $APP ok"
  done
}

warm=true
test -n "$ARG" && echo "$ARG"|grep -q "^--cold$" && warm=false
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

echo "All apps have been launched"
echo "See them on ports 5030, 5034, 5009 and 5005"
echo "Done."
