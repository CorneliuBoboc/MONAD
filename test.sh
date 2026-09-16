#! /bin/bash

set -e

APP="$1"

test -n "$APP"
test -n "$DEMO"
test -n "$VER"

cd ./projects/$APP

if echo "$APP"|grep -qv "^bfc$"; then
  $DEMO || command -v ffmpeg  || brew install ffmpeg 
#  $DEMO && command -v ffmpeg  || sudo apt update && sudo apt install -y ffmpeg 
fi
pip install -r requirements.txt 
  pip uninstall -y google-genai google-generativeai google-auth google-auth-oauthlib
  pip install -U google-genai google-auth
if echo "$APP"|grep -q "^diarix$"; then
  $DEMO || command -v whispermlx  || pip install whispermlx 
fi

python$VER app.py & pid=$!
sleep 8
test -d /proc/$pid

