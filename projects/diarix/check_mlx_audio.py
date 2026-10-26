"""Check the pinned WhisperMLX/TorchCodec stack and decode a generated WAV."""

import importlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import wave
from importlib.metadata import version
from pathlib import Path


EXPECTED_VERSIONS = {
    "whispermlx": "3.14.0",
    "pyannote.audio": "4.0.7",
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "torchvision": "0.23.0",
    "torchcodec": "0.7.0",
}


def redact_paths(message):
    for private_root in (str(Path.home()), tempfile.gettempdir()):
        private_root = private_root.rstrip("/")
        if private_root:
            message = message.replace(private_root, "<PRIVATE_PATH>")
    message = re.sub(r"(?<![\w])/(?:Users|home)/[^/\s:'\"]+", "<HOME>", message)
    message = re.sub(
        r"(?<![\w])/(?:private/)?var/folders/[^/\s]+/[^/\s]+/T(?:/[^/\s:'\"]*)?",
        "<TMP>",
        message,
    )
    return message


def main():
    print(
        f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}",
        flush=True,
    )

    for package, expected in EXPECTED_VERSIONS.items():
        installed = version(package)
        print(f"{package}=={installed}", flush=True)
        if installed != expected:
            raise RuntimeError(
                f"Expected {package}=={expected}, but found {installed}."
            )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("FFmpeg is not available on PATH.")

    ffmpeg = subprocess.run(
        [ffmpeg_path, "-version"],
        capture_output=True,
        check=True,
        text=True,
    )
    version_match = re.match(r"ffmpeg version (\S+)", ffmpeg.stdout)
    ffmpeg_version = version_match.group(1) if version_match else "unknown"
    print(f"FFmpeg version: {ffmpeg_version}", flush=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"torchcodec is not installed correctly",
            category=UserWarning,
        )
        importlib.import_module("pyannote.audio.core.io")

    from torchcodec.decoders import AudioDecoder

    with tempfile.TemporaryDirectory(prefix="diarix-audio-check-") as temp_dir:
        wav_path = Path(temp_dir) / "check.wav"
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=16000:cl=mono",
                    "-t",
                    "1",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ],
                capture_output=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(
                f"FFmpeg failed to generate the WAV test file: {detail}"
            ) from exc
        with wave.open(str(wav_path), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getsampwidth() != 2
                or wav_file.getframerate() != 16000
                or wav_file.getnframes() == 0
            ):
                raise RuntimeError("FFmpeg generated a WAV with unexpected audio parameters.")

        samples = AudioDecoder(str(wav_path)).get_all_samples()
        if samples.data.numel() == 0:
            raise RuntimeError("TorchCodec decoded the test WAV but returned no samples.")

    print("TorchCodec WAV decode: OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"Preflight failed: {type(exc).__name__}: {redact_paths(str(exc))}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
