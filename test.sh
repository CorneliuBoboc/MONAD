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
python$VER -m pip install -r requirements.txt
if echo "$APP"|grep -q "^diarix$"; then
  export DIARIX_MLX_WORKERS=4
  $DEMO || command -v whispermlx  || pip install whispermlx 
fi

python$VER app.py & pid=$!
sleep 8
test -d /proc/$pid
