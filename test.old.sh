#! /bin/bash

set -e

APP="$1"

test -n "$APP"
test -n "$DEMO"
test -n "$VER"

cd ./projects/$APP

python$VER app.py & pid=$!
sleep 8
kill -0 "$pid"
