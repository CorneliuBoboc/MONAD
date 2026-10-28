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
pip install "pyannote-audio>=4.0.0" "torch==2.8.*" "torchaudio==2.8.*" "torchcodec==0.7.*" "torchvision==0.23.*"
if echo "$APP"|grep -q "^diarix$"; then
  export DIARIX_MLX_WORKERS=1
  $DEMO || command -v whispermlx || pip install "whispermlx==3.14.0" "pyannote-audio>=4.0.0" "torch==2.8.*" "torchaudio==2.8.*" "torchcodec==0.7.*" "torchvision==0.23.*"
fi

python$VER app.py & pid=$!
sleep 8
test -d /proc/$pid
