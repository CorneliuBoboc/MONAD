"""Check the pinned WhisperMLX/TorchCodec stack and decode a generated WAV."""

import importlib
import platform
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

ffmpeg = subprocess.run(
    ["ffmpeg", "-version"],
    capture_output=True,
    check=True,
    text=True,
)
print(ffmpeg.stdout.splitlines()[0], flush=True)

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
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0" * 16000)

    samples = AudioDecoder(str(wav_path)).get_all_samples()
    if samples.data.numel() == 0:
        raise RuntimeError("TorchCodec decoded the test WAV but returned no samples.")

print("TorchCodec WAV decode: OK", flush=True)
