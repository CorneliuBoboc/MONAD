#! /bin/bash

set -e

APP="$1"

test -n "$APP"
test -n "$DEMO"
test -n "$VER"

if [ ! -d "./projects/$APP" ]; then
  echo "Unknown app selection." >&2
  exit 1
fi
cd "./projects/$APP" 2>/dev/null

if echo "$APP"|grep -qv "^morphix$"; then
  if ! $DEMO && ! command -v ffmpeg >/dev/null 2>&1; then
    if ! brew install ffmpeg >/dev/null 2>&1; then
      echo "FFmpeg installation failed." >&2
      exit 1
    fi
  fi
#  $DEMO && command -v ffmpeg  || sudo apt update && sudo apt install -y ffmpeg 
fi
if [ "$APP" = "diarix" ] && ! $DEMO; then
  export DIARIX_MLX_WORKERS=1
  echo "Diarix MLX dependency pins: whispermlx=3.14.0 pyannote.audio=4.0.7 torch=2.8.0 torchaudio=2.8.0 torchvision=0.23.0 torchcodec=0.7.0"
  if ! python$VER -m pip install -r requirements.txt -r requirements-mlx.txt >/dev/null 2>&1; then
    echo "Dependency installation failed for Diarix." >&2
    exit 1
  fi
  if ! python$VER -m pip check >/dev/null 2>&1; then
    echo "Dependency consistency check failed for Diarix." >&2
    exit 1
  fi
  if ! python$VER check_mlx_audio.py; then
    echo "Diarix audio runtime preflight failed." >&2
    exit 1
  fi
else
  if ! python$VER -m pip install -r requirements.txt >/dev/null 2>&1; then
    echo "Dependency installation failed for $APP." >&2
    exit 1
  fi
fi

python$VER app.py >/dev/null 2>&1 & pid=$!
sleep 8
if ! kill -0 "$pid" 2>/dev/null; then
  echo "$APP failed its startup check." >&2
  exit 1
fi
echo "$APP startup check passed."
