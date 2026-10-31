#! /bin/bash

set -e

APPS="morphix diarix"
ARG="$1"
OS="$(uname)"
PORTS="5029 5030 5034"
VER="3.12"

DEMO=false
echo "$OS"|grep -q "^Linux$" && DEMO=true
$DEMO && VER="3.12"
#$DEMO && APPS=diarix
ENGINE=mlx
$DEMO && ENGINE=faster
$DEMO || DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib

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
  SCRIPT="test.old.sh"
pip install "annotated-types==0.8.0" "anyio==4.15.1" "beautifulsoup4==4.15.0" "blinker==1.9.0" "certifi==2026.10.23" "cffi==2.1.1" "charset-normalizer==3.5.2" "click==8.5.0" "cryptography==50.0.2" "distro==1.9.0" "EbookLib==0.20" "Flask==3.1.3" "flatbuffers==25.12.19" "google-auth==2.62.0" "google-genai==2.34.0" "h11==0.16.0" "httpcore==1.0.9" "httpx==0.28.1" "idna==3.20" "itsdangerous==2.2.0" "Jinja2==3.1.6" "lxml==6.1.3" "MarkupSafe==3.0.4" "networkx==3.7" "numpy==2.5.4" "onnxruntime==1.31.1" "packaging==26.3" "pillow==13.0.0" "protobuf==7.36.2" "psutil==7.2.2" "pyasn1==0.6.5" "pyasn1_modules==0.4.2" "pycparser==3.12" "pydantic==2.14.2" "pydantic_core==2.50.2" "pymupdf4llm==2.0" "pymupdf==2.0" "pymupdf_layout==2.0" "pypdf==6.21.4" "python-docx==1.2.0" "PyYAML==6.0.3" "reportlab==5.1.0" "requests==2.34.2" "six==1.17.0" "sniffio==1.3.1" "soupsieve==3.0.4" "tabulate==0.10.0" "tenacity==9.1.4" "typing-inspection==0.4.4" "typing_extensions==4.16.0" "urllib3==2.8.0" "websockets==17.4" "Werkzeug==3.1.9" "yt-dlp==2026.10.26"
if [ $ENGINE == faster ]; then
  pip install "faster-whisper==1.2.1" "av==18.1.0" "pyannote.audio"
else
  pip install "whispermlx==3.14.0" "pyannote.audio==4.0.7" "torch==2.8.0" "torchaudio==2.8.0" "torchcodec==0.7.0" "torchvision==0.23.0"
fi
  pip check
  source .venv/bin/activate
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
echo "See them on ports 5030, 5034"
echo "Done."
