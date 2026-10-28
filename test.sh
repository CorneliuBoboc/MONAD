#! /bin/bash

set -e

APP="$1"

test -n "$APP"
test -n "$DEMO"
test -n "$VER"

cd ./projects/$APP

if echo "$APP"|grep -qv "^morphix$"; then
  $DEMO || command -v ffmpeg  || brew install ffmpeg 
#  $DEMO && command -v ffmpeg  || sudo apt update && sudo apt install -y ffmpeg 
fi
(cd ../..; resolve_install.sh projects/$APP/requirements.txt .venv)
if echo "$APP"|grep -q "^diarix$"; then
  export DIARIX_MLX_WORKERS=1
fi

python$VER app.py & pid=$!
sleep 8
test -d /proc/$pid
