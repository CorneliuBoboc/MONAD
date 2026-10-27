#!/usr/bin/env bash
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    printf 'Usage: %s INPUT [OUTPUT.wav]\n' "$0" >&2
    exit 2
fi

input=$1
output=${2:-"${input%.*}_16k_mono.wav"}

# Extract the first audio stream and normalize it to 16 kHz mono 16-bit PCM WAV.
ffmpeg -hide_banner -y -i "$input" \
    -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le \
    "$output"
