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
if [ "$APP" = "diarix" ] && ! $DEMO; then
  export DIARIX_MLX_WORKERS=1
  python$VER -m pip install -r requirements.txt -r requirements-mlx.txt
  python$VER -m pip check
  python$VER check_mlx_audio.py
else
  python$VER -m pip install -r requirements.txt
fi

python$VER app.py & pid=$!
sleep 8
test -d /proc/$pid
