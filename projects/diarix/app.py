""" python
Media Editor — Flask app
Load media (upload or URL) -> preview & cut chunks with ffmpeg -> transcribe
(openai-whisper / faster-whisper / whispermlx) with optional speaker
diarization (pyannote.audio or an LLM-based heuristic via Claude/Gemini) ->
download the transcript and/or a zip of everything.

Run:
    pip install -r requirements.txt
    python app.py
"""

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from urllib.parse import urlparse
from uuid import uuid4

import requests
from flask import (
    Flask, request, jsonify, send_file, abort, render_template_string
)
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.join(tempfile.gettempdir(), "media_editor_sessions")

# In-memory workspace registry: sid -> {dir, source, chunks, transcript, segments, ...}
SESSIONS = {}
ACTIVE_SID = None
_SESS_LOCK = threading.RLock()

# GPU work is serialized by default (single permitted thread). Override via
# DIARIX_GPU_WORKERS only as an experimental escape hatch; values <1 or
# invalid fall back to 1. DIARIX_FORCE_SEQUENTIAL=1 disables all pools.
GPU_DEVICES = {"cuda", "metal"}


def gpu_worker_count():
    """GPU concurrency. Default and hard fallback: a single thread."""
    raw = os.environ.get("DIARIX_GPU_WORKERS", "1").strip()
    try:
        workers = int(raw)
    except ValueError:
        return 1
    if workers < 1:
        return 1
    # Cap experimental multi-GPU-thread attempts.
    return min(workers, 4)


_GPU_GATE = threading.Semaphore(gpu_worker_count())

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

# Languages we explicitly accept. Anything else (or "auto") triggers Whisper's
# built-in language detection.
VALID_LANGUAGES = {"auto", "de", "en", "fr", "ro"}


# --------------------------------------------------------------------------
# workspace / filesystem helpers
# --------------------------------------------------------------------------

def _empty_transcription_status():
    return {"state": "idle", "completed": 0, "total": 0, "error": None}


def _new_session_dict(sid):
    d = os.path.join(BASE_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return {
        "dir": d,
        "source": None,
        "chunks": [],
        "transcript": None,
        "segments": [],
        "source_files": [],   # list of loaded files (multi-file support)
        "multifile": False,   # True only for multi-disk uploads
        "transcriptions": [], # list of per-file transcriptions (multi-file mode)
        "transcription_status": _empty_transcription_status(),
        "lock": threading.RLock(),
    }


def purge_all_session_dirs():
    """Remove leftover session dirs from previous server processes."""
    global ACTIVE_SID
    with _SESS_LOCK:
        SESSIONS.clear()
        ACTIVE_SID = None
        if os.path.isdir(BASE_DIR):
            shutil.rmtree(BASE_DIR, ignore_errors=True)
        os.makedirs(BASE_DIR, exist_ok=True)


# Fresh disk state on every process start — no stale audio from prior runs.
purge_all_session_dirs()


def ensure_sid():
    global ACTIVE_SID
    with _SESS_LOCK:
        if ACTIVE_SID is None:
            ACTIVE_SID = uuid4().hex
        return ACTIVE_SID


def get_sid_or_none():
    return ACTIVE_SID


def get_session(sid, create=False):
    if not sid:
        return None
    with _SESS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None and create:
            sess = _new_session_dict(sid)
            SESSIONS[sid] = sess
        return sess


def serialize_current_session(view):
    """Keep session file operations atomic against transcription and other edits."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        sid = ensure_sid()
        sess = get_session(sid, create=True)
        with sess["lock"]:
            return view(*args, _session=sess, _sid=sid, **kwargs)
    return wrapped


def wipe_session_dir(sess):
    if os.path.isdir(sess["dir"]):
        shutil.rmtree(sess["dir"], ignore_errors=True)
    os.makedirs(sess["dir"], exist_ok=True)
    sess["source"] = None
    sess["chunks"] = []
    sess["transcript"] = None
    sess["segments"] = []
    sess["source_files"] = []
    sess["multifile"] = False
    sess["transcriptions"] = []
    sess["transcription_status"] = _empty_transcription_status()


def public_file_entry(f):
    return {
        "id": f["id"],
        "name": f["name"],
        "kind": f["kind"],
        "duration": f["duration"],
        "file_size": f["file_size"],
    }


def sync_primary_source(sess):
    """Keep sess['source'] / multifile flags aligned with source_files order."""
    files = sess["source_files"]
    sess["multifile"] = len(files) > 1
    if not files:
        sess["source"] = None
        return
    first = files[0]
    sess["source"] = {
        "path": first["path"],
        "ext": first["ext"],
        "kind": first["kind"],
        "duration": first["duration"],
        "file_size": first["file_size"],
        "format": first["format"],
        "video_info": first["video_info"],
        "audio_info": first["audio_info"],
    }


def clear_file_pool(sess):
    """Clear loaded sources, chunks, and results without removing session state."""
    paths = [f.get("path") for f in sess.get("source_files", [])]
    paths.extend(c.get("path") for c in sess.get("chunks", []))
    failures = []
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            app.logger.warning("Could not remove media file %s", path, exc_info=True)
            failures.append(path)

    if failures:
        return len(failures)
    sess["chunks"] = []
    sess["transcriptions"] = []
    sess["transcript"] = None
    sess["segments"] = []
    sess["transcription_status"] = _empty_transcription_status()
    sess["source_files"] = []
    sync_primary_source(sess)
    return 0


def files_payload(sess, sid):
    files = sess["source_files"]
    if not files:
        return {
            "kind": None, "duration": 0, "src": None, "file_size": 0,
            "format": None, "video_info": None, "audio_info": None,
            "multifile": False, "file_count": 0, "files": [],
        }
    first = files[0]
    return {
        "kind": first["kind"],
        "duration": first["duration"],
        "src": f"/media/{sid}/source",
        "file_size": first["file_size"],
        "format": first["format"],
        "video_info": first["video_info"],
        "audio_info": first["audio_info"],
        "multifile": sess["multifile"],
        "file_count": len(files),
        "files": [public_file_entry(f) for f in files],
    }


def force_sequential():
    return os.environ.get("DIARIX_FORCE_SEQUENTIAL", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def cpu_worker_count(n_tasks):
    """How many CPU workers to use; always at least 1, never above n_tasks."""
    n_tasks = max(1, int(n_tasks))
    raw = os.environ.get("DIARIX_CPU_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), n_tasks))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 2, n_tasks, 8))


def uses_gpu_device(engine, device):
    if engine == "mlx":
        return True
    return (device or "").lower() in GPU_DEVICES


def run_cpu_jobs(jobs, worker_count=None):
    """Run callables in parallel on CPU. Each job is a zero-arg callable.

    Returns a list of results in the same order as `jobs`.
    Falls back to sequential execution on pool failure.
    """
    if not jobs:
        return []
    if force_sequential() or len(jobs) == 1:
        return [job() for job in jobs]

    workers = worker_count if worker_count is not None else cpu_worker_count(len(jobs))
    workers = max(1, min(workers, len(jobs)))
    if workers == 1:
        return [job() for job in jobs]

    try:
        results = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job): i for i, job in enumerate(jobs)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results
    except Exception as exc:
        app.logger.warning("CPU pool failed (%s); falling back to sequential.", exc)
        return [job() for job in jobs]


def run_gpu_exclusive(fn):
    """Run fn holding the process-wide GPU gate (single permitted thread)."""
    acquired = _GPU_GATE.acquire(timeout=3600)
    if not acquired:
        raise RuntimeError("Timed out waiting for the GPU transcription slot.")
    try:
        return fn()
    finally:
        _GPU_GATE.release()


def fmt_ts(t):
    t = max(0.0, float(t))
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


def normalize_language(lang):
    """Return a Whisper-compatible language value.

    'auto' (or empty/unknown) -> None, which tells Whisper to auto-detect.
    """
    if not lang or lang == "auto" or lang not in VALID_LANGUAGES:
        return None
    return lang


# --------------------------------------------------------------------------
# media probing / cutting (ffmpeg / ffprobe)
# --------------------------------------------------------------------------

def probe_media(path):
    """Return {'kind': 'video'|'audio', 'duration': float, ...} or None on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, check=True, timeout=30,
        )
        data = json.loads(out.stdout.decode("utf-8", errors="ignore"))
    except Exception:
        return None

    duration = None
    fmt = data.get("format", {})
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except ValueError:
            duration = None

    has_video = False
    video_info = {}
    audio_info = {}

    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and s.get("codec_name") not in ("mjpeg", "png", "bmp"):
            has_video = True
            video_info = {
                "codec": s.get("codec_name", "unknown"),
                "width": s.get("width"),
                "height": s.get("height"),
                "fps": s.get("r_frame_rate"),
                "bitrate": s.get("bit_rate")
            }
        if duration is None and s.get("duration"):
            try:
                duration = max(duration or 0.0, float(s["duration"]))
            except ValueError:
                pass
        if s.get("codec_type") == "audio":
            audio_info = {
                "codec": s.get("codec_name", "unknown"),
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
                "bitrate": s.get("bit_rate")
            }

    if duration is None:
        return None

    file_size = os.path.getsize(path)
    format_name = fmt.get("format_name", "unknown")

    return {
        "kind": "video" if has_video else "audio",
        "duration": duration,
        "file_size": file_size,
        "format": format_name,
        "video_info": video_info if has_video else None,
        "audio_info": audio_info
    }


def cut_chunk(source_path, kind, start, end, out_path):
    cmd = ["ffmpeg", "-y", "-i", source_path, "-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    if kind == "video":
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart"]
    else:
        cmd += ["-vn", "-c:a", "aac"]
    cmd += [out_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)


# --------------------------------------------------------------------------
# transcription engines
# --------------------------------------------------------------------------

def run_whisper_openai(path, language, device, model_size):
    try:
        import whisper
    except ImportError:
        raise RuntimeError("openai-whisper is not installed on the server (pip install openai-whisper).")
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(path, language=language, verbose=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]


def run_faster_whisper(path, language, device, model_size):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper is not installed on the server (pip install faster-whisper).")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(path, language=language)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def run_mlx_whisper_cli(path, language, model_size, hf_token=None, diarize=False):
    """Use whispermlx CLI tool instead of Python library.

    This matches the usage pattern from test.sh and provides native diarization support.
    `language` may be None (auto-detect) or a 2-letter code.
    """
    if not shutil.which("whispermlx"):
        raise RuntimeError(
            "whispermlx command not found. Please install it: pip install whispermlx"
        )

    # whispermlx expects a literal "auto" when no language is fixed.
    lang_arg = language if language else "auto"

    cmd = [
        "whispermlx",
        path,
        "--model", model_size,
        "--language", lang_arg,
        "--output_format", "json",
        "--output_dir", os.path.dirname(path),
        "--condition_on_previous_text", "False",
        "--compression_ratio_threshold", "2.4",
        "--logprob_threshold", "-1.0",
        "--no_speech_threshold", "0.6",
    ]

    if diarize:
        if not hf_token:
            raise RuntimeError(
                "Hugging Face token required for diarization with whispermlx. "
                "Please provide it in the API key field."
            )
        cmd.extend(["--diarize", "--hf_token", hf_token])

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            check=True
        )

        base_name = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(os.path.dirname(path), f"{base_name}.json")

        if not os.path.exists(json_path):
            for f in os.listdir(os.path.dirname(path)):
                if f.endswith(".json") and base_name in f:
                    json_path = os.path.join(os.path.dirname(path), f)
                    break

        if not os.path.exists(json_path):
            raise RuntimeError(
                f"whispermlx completed but output JSON file not found for {path}"
            )

        with open(json_path, 'r') as f:
            data = json.load(f)

        segments = []
        if "segments" in data:
            for seg in data["segments"]:
                segment = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                }
                if "speaker" in seg:
                    segment["speaker"] = seg["speaker"]
                segments.append(segment)
        elif "text" in data:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": data["text"].strip()
            })

        return segments

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        raise RuntimeError(
            f"whispermlx failed with error:\n{error_msg[:500]}"
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse whispermlx output: {e}")


def _transcribe_file_unlocked(path, engine, language, device, model_size, hf_token=None, diarize=False):
    # Normalize "auto" and unknown codes to None for the Python engines.
    lang = normalize_language(language)
    if engine == "openai-whisper":
        return run_whisper_openai(path, lang, device, model_size)
    if engine == "faster":
        return run_faster_whisper(path, lang, device, model_size)
    if engine == "mlx":
        # MLX CLI takes an explicit "auto" string (handled inside the runner).
        return run_mlx_whisper_cli(path, lang, model_size, hf_token, diarize)
    raise RuntimeError(f"Unknown transcription engine '{engine}'.")


def transcribe_file(path, engine, language, device, model_size, hf_token=None, diarize=False):
    """Transcribe one file. GPU/Metal/MLX calls are serialized process-wide."""
    def _run():
        return _transcribe_file_unlocked(
            path, engine, language, device, model_size, hf_token=hf_token, diarize=diarize,
        )

    if uses_gpu_device(engine, device):
        # Experimental multi-worker GPU still funnels through the gate so the
        # hardware sees one job at a time unless the semaphore is widened.
        return run_gpu_exclusive(_run)
    return _run()


def normalize_segments(segs):
    norm_segs = []
    for s in segs or []:
        seg = {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
        }
        if s.get("speaker"):
            seg["speaker"] = s["speaker"]
        norm_segs.append(seg)
    return norm_segs


def segments_to_transcript(norm_segs):
    lines = []
    for seg in norm_segs:
        prefix = f"{seg['speaker']}: " if seg.get("speaker") else ""
        lines.append(f"[{fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}] {prefix}{seg['text']}")
    return "\n".join(lines)


def apply_diarization_to_segments(
    path, norm_segs, diarize, diarize_method, ai_provider, api_key, num_speakers,
):
    if not diarize or any(s.get("speaker") for s in norm_segs):
        return norm_segs
    if diarize_method == "pyannote":
        turns = run_pyannote(path, api_key, num_speakers)
        assign_speakers_from_turns(norm_segs, turns)
    elif diarize_method == "ai_api":
        labels = run_ai_diarization(norm_segs, ai_provider, api_key, num_speakers)
        for seg, label in zip(norm_segs, labels):
            seg["speaker"] = label
    else:
        raise RuntimeError("Choose a diarization method (pyannote or AI API).")
    return norm_segs


def transcribe_one_source_file(
    f, engine, language, device, model_size,
    diarize, diarize_method, ai_provider, api_key, num_speakers,
):
    """Full per-file pipeline used by multi-file mode."""
    segs = transcribe_file(
        f["path"], engine, language, device, model_size,
        hf_token=api_key, diarize=False,
    )
    norm_segs = normalize_segments(segs)
    apply_diarization_to_segments(
        f["path"], norm_segs, diarize, diarize_method, ai_provider, api_key, num_speakers,
    )
    text = segments_to_transcript(norm_segs)
    return {"name": f["name"], "id": f.get("id"), "transcript": text, "segments": norm_segs}


def publish_ready_transcriptions(sess, slots, publish_lock):
    """Expose every finished file, ordered by its position in the source list."""
    with publish_lock, sess["lock"]:
        results = [item for item in slots if item is not None]
        sess["transcriptions"] = results
        # Multi-file results remain independent; never create a combined transcript.
        sess["transcript"] = None
        sess["transcription_status"]["completed"] = len(results)
        return list(results)


def run_multifile_transcriptions(
    sess, source_files, engine, language, device, model_size,
    diarize, diarize_method, ai_provider, api_key, num_speakers,
):
    """Transcribe many files with CPU parallelism / GPU serialization.

    Results are published one-by-one in the user-established file order, even
    when workers finish out of order.
    """
    n = len(source_files)
    slots = [None] * n
    publish_lock = threading.Lock()
    first_error = []

    def work(idx, f):
        try:
            return idx, transcribe_one_source_file(
                f, engine, language, device, model_size,
                diarize, diarize_method, ai_provider, api_key, num_speakers,
            ), None
        except Exception as exc:
            return idx, None, exc

    def consume(idx, result, exc):
        if exc is not None:
            first_error.append(exc)
            return
        slots[idx] = result
        return publish_ready_transcriptions(sess, slots, publish_lock)

    def run_sequential():
        for idx, f in enumerate(source_files):
            idx2, result, exc = work(idx, f)
            consume(idx2, result, exc)
        if first_error:
            raise first_error[0]
        return list(sess["transcriptions"])

    on_gpu = uses_gpu_device(engine, device)

    if force_sequential() or n <= 1:
        return run_sequential()

    if on_gpu:
        workers = gpu_worker_count()
        if workers <= 1:
            return run_sequential()
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, i, f) for i, f in enumerate(source_files)]
                for fut in as_completed(futures):
                    idx, result, exc = fut.result()
                    consume(idx, result, exc)
            if first_error:
                raise first_error[0]
            return list(sess["transcriptions"])
        except Exception as exc:
            if first_error:
                raise first_error[0]
            app.logger.warning(
                "GPU pool path failed (%s); falling back to sequential.", exc
            )
            with sess["lock"]:
                sess["transcriptions"] = []
                sess["transcription_status"]["completed"] = 0
            for i in range(n):
                slots[i] = None
            return run_sequential()

    workers = cpu_worker_count(n)
    if workers <= 1:
        return run_sequential()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i, f) for i, f in enumerate(source_files)]
            for fut in as_completed(futures):
                idx, result, exc = fut.result()
                consume(idx, result, exc)
        if first_error:
            raise first_error[0]
        return list(sess["transcriptions"])
    except Exception as exc:
        if first_error:
            raise first_error[0]
        app.logger.warning(
            "CPU transcription pool failed (%s); sequential fallback.", exc
        )
        with sess["lock"]:
            sess["transcriptions"] = []
            sess["transcription_status"]["completed"] = 0
        for i in range(n):
            slots[i] = None
        return run_sequential()


# --------------------------------------------------------------------------
# diarization
# --------------------------------------------------------------------------

def run_pyannote(path, hf_token, num_speakers=None):
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError("pyannote.audio is not installed on the server (pip install pyannote.audio).")
    if not hf_token:
        raise RuntimeError("A Hugging Face access token is required for pyannote diarization.")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
    diarization = pipeline(path, num_speakers=num_speakers) if num_speakers else pipeline(path)

    turns = []
    for turn, speaker in diarization.speaker_diarization:
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns


def assign_speakers_from_turns(segments, turns):
    if not turns:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return
    for seg in segments:
        best, best_overlap = None, 0.0
        for t in turns:
            overlap = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, t
        if best is None:
            best = min(turns, key=lambda t: abs(t["start"] - seg["start"]))
        seg["speaker"] = best["speaker"]


def run_ai_diarization(segments, provider, api_key, num_speakers=None):
    """Heuristic, text-based speaker labeling via an LLM (no separate audio model)."""
    if not api_key:
        raise RuntimeError("An API key is required for AI-based diarization.")

    payload = [{"i": i, "start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
               for i, s in enumerate(segments)]

    constraint = ""
    if num_speakers:
        constraint = f"There are exactly {num_speakers} speakers. Label them SPEAKER_00, SPEAKER_01, ... "
    else:
        constraint = "Infer a reasonable number of distinct speakers from context and label them SPEAKER_00, SPEAKER_01, ..."

    prompt = (
        "You are labeling speaker turns in a transcript based only on the text and timing below "
        "(no audio). " + constraint + " Respond with ONLY a JSON array of objects. Return exactly "
        "one object for every input segment; do not merge, omit, duplicate, or reorder segments. "
        "Each object must have the input index 'i' and a 'label', for example "
        "[{\"i\": 0, \"label\": \"SPEAKER_00\"}].\n\n"
        f"Segments: {json.dumps(payload)}"
    )

    if provider == "claude":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("The 'anthropic' package is not installed on the server (pip install anthropic).")
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    elif provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("The 'google-genai' package is not installed on the server (pip install google-genai).")
        client = genai.Client(api_key=api_key)
        text = ""
        finish_reason = None
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                request_prompt += (
                    "\n\nYour previous response was incomplete or invalid. Return the complete "
                    "JSON array again, including every input index from 0 through "
                    f"{len(segments) - 1}."
                )
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=request_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=8000,
                    response_mime_type="application/json",
                ),
            )
            text = response.text or ""
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            data = None
            try:
                match = re.search(r"\[.*\]", text, re.DOTALL)
                data = json.loads(match.group(0)) if match else None
                labels = [item["label"] for item in data]
                indices = [item["i"] for item in data]
                if (
                    len(labels) == len(segments)
                    and indices == list(range(len(segments)))
                    and all(isinstance(label, str) and label for label in labels)
                ):
                    return labels
            except (TypeError, KeyError, IndexError, json.JSONDecodeError):
                pass
            app.logger.warning(
                "Gemini diarization response incomplete: attempt=%d segments=%d "
                "labels=%d finish_reason=%s",
                attempt + 1,
                len(segments),
                len(data) if isinstance(data, list) else 0,
                finish_reason,
            )
        raise RuntimeError("The AI diarization response did not label every segment.")
    else:
        raise RuntimeError(f"Unknown AI provider '{provider}'.")

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise RuntimeError("The AI diarization response could not be parsed as JSON.")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise RuntimeError("The AI diarization response was not a JSON array.")
    try:
        labels = [item["label"] for item in data]
        indices = [item["i"] for item in data]
    except (TypeError, KeyError):
        raise RuntimeError("The AI diarization response had an invalid label format.")
    if (
        len(labels) != len(segments)
        or indices != list(range(len(segments)))
        or not all(isinstance(label, str) and label for label in labels)
    ):
        raise RuntimeError("The AI diarization response did not label every segment.")
    return labels


# --------------------------------------------------------------------------
# routes — pages / media
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/media/<sid>/source")
def media_source(sid):
    sess = SESSIONS.get(sid)
    if not sess or not sess.get("source"):
        abort(404)
    return send_file(sess["source"]["path"], conditional=True)


# --------------------------------------------------------------------------
# routes — API
# --------------------------------------------------------------------------

@app.route("/api/session/reset", methods=["POST"])
def api_session_reset():
    """Start a clean workspace — drops any leftover audio from prior use."""
    global ACTIVE_SID
    with _SESS_LOCK:
        old = ACTIVE_SID
        old_sess = SESSIONS.get(old) if old else None
        if old_sess:
            with old_sess["lock"]:
                if old_sess["transcription_status"]["state"] == "running":
                    return jsonify(error="Wait for the current transcription to finish before resetting."), 409
        if old and old in SESSIONS:
            old_sess = SESSIONS.pop(old)
            try:
                if os.path.isdir(old_sess["dir"]):
                    shutil.rmtree(old_sess["dir"], ignore_errors=True)
            except Exception:
                pass
        # Also sweep any orphan dirs left behind.
        if os.path.isdir(BASE_DIR):
            for name in os.listdir(BASE_DIR):
                path = os.path.join(BASE_DIR, name)
                if os.path.isdir(path) and name not in SESSIONS:
                    shutil.rmtree(path, ignore_errors=True)
        ACTIVE_SID = uuid4().hex
        sess = _new_session_dict(ACTIVE_SID)
        SESSIONS[ACTIVE_SID] = sess
        return jsonify(ok=True, sid=ACTIVE_SID, files=[], file_count=0)


@app.route("/api/files", methods=["GET"])
def api_list_files():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(files=[], file_count=0, multifile=False)
    return jsonify(files_payload(sess, sid))


@app.route("/api/files", methods=["DELETE"])
def api_clear_files():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(error="No session."), 400
    with sess["lock"]:
        if sess["transcription_status"]["state"] == "running":
            return jsonify(error="Wait for the current transcription to finish before changing files."), 409
        failed_removals = clear_file_pool(sess)
        if failed_removals:
            return jsonify(
                error=f"Could not remove {failed_removals} media file(s). The list was kept; retry after checking file permissions."
            ), 500
        return jsonify(files_payload(sess, sid))


@app.route("/api/files/<file_id>", methods=["DELETE"])
def api_delete_file(file_id):
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(error="No session."), 400
    with sess["lock"]:
        if sess["transcription_status"]["state"] == "running":
            return jsonify(error="Wait for the current transcription to finish before changing files."), 409

        keep, drop = [], []
        for f in sess["source_files"]:
            (drop if f.get("id") == file_id else keep).append(f)
        if not drop:
            return jsonify(error="File not found."), 404

        for f in drop:
            try:
                os.remove(f["path"])
            except FileNotFoundError:
                pass
            except OSError as exc:
                app.logger.warning("Could not remove media file %s", f["path"], exc_info=True)
                return jsonify(error=f"Could not remove file: {exc}"), 500

        sess["source_files"] = keep
        sess["chunks"] = []
        sess["transcriptions"] = []
        sess["transcript"] = None
        sess["segments"] = []
        sess["transcription_status"] = _empty_transcription_status()
        sync_primary_source(sess)
        return jsonify(files_payload(sess, sid))


@app.route("/api/files/<file_id>/rename", methods=["POST"])
def api_rename_file(file_id):
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(error="No session."), 400
    with sess["lock"]:
        if sess["transcription_status"]["state"] == "running":
            return jsonify(error="Wait for the current transcription to finish before changing files."), 409

        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name")
        if not isinstance(name, str):
            return jsonify(error="Provide a file name."), 400
        name = secure_filename(name.strip())
        if not name:
            return jsonify(error="Provide a valid file name."), 400

        source_file = next((f for f in sess["source_files"] if f.get("id") == file_id), None)
        if source_file is None:
            return jsonify(error="File not found."), 404
        if not os.path.splitext(name)[1]:
            name += source_file.get("ext") or ""
        source_file["name"] = name
        sess["transcriptions"] = []
        sess["transcript"] = None
        sess["segments"] = []
        sess["transcription_status"] = _empty_transcription_status()
        return jsonify(files_payload(sess, sid))


@app.route("/api/files/reorder", methods=["POST"])
def api_reorder_files():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(error="No session."), 400
    with sess["lock"]:
        if sess["transcription_status"]["state"] == "running":
            return jsonify(error="Wait for the current transcription to finish before changing files."), 409

        data = request.get_json(force=True, silent=True) or {}
        order = data.get("order")
        if not isinstance(order, list) or not order:
            return jsonify(error="Provide an 'order' list of file ids."), 400

        by_id = {f["id"]: f for f in sess["source_files"]}
        if (not all(isinstance(file_id, str) for file_id in order)
                or set(order) != set(by_id.keys()) or len(order) != len(by_id)):
            return jsonify(error="Order must list each current file id exactly once."), 400

        sess["source_files"] = [by_id[i] for i in order]
        sess["transcriptions"] = []
        sess["transcript"] = None
        sess["segments"] = []
        sess["transcription_status"] = _empty_transcription_status()
        sync_primary_source(sess)
        return jsonify(files_payload(sess, sid))


@app.route("/api/load", methods=["POST"])
@serialize_current_session
def api_load(_session=None, _sid=None):
    sid = _sid or ensure_sid()
    sess = _session or get_session(sid, create=True)
    if sess["transcription_status"]["state"] == "running":
        return jsonify(error="Wait for the current transcription to finish before adding files."), 409

    up_list = request.files.getlist("files") or []
    single = request.files.get("file")
    if not up_list and single and single.filename:
        up_list = [single]
    up_list = [u for u in up_list if u and u.filename]

    url = (request.form.get("url") or "").strip()

    if up_list:
        # ---- additive disk-upload path ----
        files_dir = os.path.join(sess["dir"], "sources")
        os.makedirs(files_dir, exist_ok=True)

        # Save uploads first (I/O), then probe in parallel on CPU.
        saved = []
        start_index = len(sess["source_files"])
        for i, up in enumerate(up_list, start=start_index):
            ext = os.path.splitext(secure_filename(up.filename))[1].lower() or ".mp4"
            file_id = uuid4().hex
            src_path = os.path.join(files_dir, f"{i:04d}_{file_id}_source{ext}")
            up.save(src_path)
            saved.append({
                "id": file_id,
                "path": src_path,
                "name": secure_filename(up.filename) or f"file_{i}{ext}",
                "ext": ext,
            })

        def _probe_one(meta):
            info = probe_media(meta["path"])
            if info is None:
                try:
                    os.remove(meta["path"])
                except OSError:
                    pass
                return None
            return {
                "id": meta["id"],
                "path": meta["path"],
                "name": meta["name"],
                "ext": meta["ext"],
                "kind": info["kind"],
                "duration": info["duration"],
                "file_size": info["file_size"],
                "format": info["format"],
                "video_info": info["video_info"],
                "audio_info": info["audio_info"],
            }

        probed = run_cpu_jobs([lambda m=m: _probe_one(m) for m in saved])
        loaded = [p for p in probed if p is not None]

        if not loaded:
            return jsonify(error="None of the newly uploaded files could be read as media."), 400

        sess["source_files"].extend(loaded)
        sess["transcriptions"] = []
        sess["transcription_status"] = _empty_transcription_status()
        # Clear chunks — they belong to the previous primary source.
        for c in sess.get("chunks") or []:
            try:
                os.remove(c["path"])
            except OSError:
                pass
        sess["chunks"] = []
        sync_primary_source(sess)

        return jsonify(**files_payload(sess, sid))

    if url:
        # ---- URL path: always single-file, multifile=False ----
        wipe_session_dir(sess)
        is_youtube = any(domain in url.lower() for domain in ['youtube.com', 'youtu.be'])
        src_path = None

        if is_youtube:
            try:
                import yt_dlp
                format_option = request.form.get('format_option', 'best')
                ydl_opts = {
                    'format': format_option,
                    'outtmpl': os.path.join(sess["dir"], 'source.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    downloaded_file = ydl.prepare_filename(info)
                    base = os.path.splitext(downloaded_file)[0]
                    for f in os.listdir(sess["dir"]):
                        if f.startswith(os.path.basename(base)):
                            src_path = os.path.join(sess["dir"], f)
                            break
                    if not src_path:
                        return jsonify(error="YouTube download completed but file not found."), 500
            except ImportError:
                return jsonify(error="yt-dlp is not installed on the server (pip install yt-dlp)."), 500
            except Exception as e:
                return jsonify(error=f"YouTube download failed: {e}"), 400
        else:
            ext = os.path.splitext(urlparse(url).path)[1].lower()
            if ext not in VIDEO_EXTS | AUDIO_EXTS:
                ext = ".mp4"
            src_path = os.path.join(sess["dir"], f"source{ext}")
            try:
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(src_path, "wb") as fh:
                        for chunk in r.iter_content(65536):
                            fh.write(chunk)
            except Exception as e:
                return jsonify(error=f"Could not download that URL ({e})."), 400

        info = probe_media(src_path)
        if info is None:
            return jsonify(error="That file doesn't look like a media file ffmpeg can read."), 400

        file_id = uuid4().hex
        sess["source_files"] = [{
            "id": file_id,
            "path": src_path,
            "name": os.path.basename(src_path),
            "ext": os.path.splitext(src_path)[1],
            "kind": info["kind"],
            "duration": info["duration"],
            "file_size": info["file_size"],
            "format": info["format"],
            "video_info": info["video_info"],
            "audio_info": info["audio_info"],
        }]
        sess["transcriptions"] = []
        sess["transcription_status"] = _empty_transcription_status()
        sync_primary_source(sess)

        return jsonify(**files_payload(sess, sid))

    return jsonify(error="Provide a file or a URL."), 400


@app.route("/api/youtube/formats", methods=["POST"])
def api_youtube_formats():
    """Get available download formats for a YouTube URL."""
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify(error="Provide a YouTube URL."), 400

    is_youtube = any(domain in url.lower() for domain in ['youtube.com', 'youtu.be'])
    if not is_youtube:
        return jsonify(error="Not a YouTube URL."), 400

    try:
        import yt_dlp

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            formats = []
            for f in info.get('formats', []):
                format_info = {
                    'format_id': f.get('format_id'),
                    'ext': f.get('ext'),
                    'resolution': f.get('resolution') or f"{f.get('width', '?')}x{f.get('height', '?')}",
                    'fps': f.get('fps'),
                    'vcodec': f.get('vcodec'),
                    'acodec': f.get('acodec'),
                    'filesize': f.get('filesize'),
                    'filesize_approx': f.get('filesize_approx'),
                    'format_note': f.get('format_note'),
                    'quality': f.get('quality'),
                }
                formats.append(format_info)

            return jsonify(formats=formats)
    except ImportError:
        return jsonify(error="yt-dlp is not installed on the server (pip install yt-dlp)."), 500
    except Exception as e:
        return jsonify(error=f"Failed to get YouTube formats: {e}"), 400


@app.route("/api/chunk", methods=["POST"])
def api_add_chunk():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess or not sess.get("source"):
        return jsonify(error="Load a media file first."), 400
    if sess.get("multifile"):
        return jsonify(error="Chunking is disabled for multi-file uploads."), 400

    data = request.get_json(force=True, silent=True) or {}
    try:
        start = float(data.get("start"))
        end = float(data.get("end"))
    except (TypeError, ValueError):
        return jsonify(error="Invalid start/end."), 400

    dur = sess["source"]["duration"]
    start = max(0.0, min(start, dur))
    end = max(0.0, min(end, dur))
    if end - start < 0.05:
        return jsonify(error="End must be at least 0.05s after start."), 400

    idx = (max((c["id"] for c in sess["chunks"]), default=0)) + 1
    kind = sess["source"]["kind"]
    out_ext = ".mp4" if kind == "video" else ".m4a"
    chunks_dir = os.path.join(sess["dir"], "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    out_path = os.path.join(chunks_dir, f"chunk_{idx}{out_ext}")

    try:
        cut_chunk(sess["source"]["path"], kind, start, end, out_path)
    except subprocess.CalledProcessError as e:
        return jsonify(error=f"ffmpeg failed: {e.stderr.decode(errors='ignore')[-400:]}"), 500

    sess["chunks"].append({
        "id": idx, "start": start, "end": end, "duration": end - start,
        "path": out_path, "filename": os.path.basename(out_path),
    })
    return jsonify(chunks=[{k: v for k, v in c.items() if k != "path"} for c in sess["chunks"]])


@app.route("/api/chunk/<int:cid>", methods=["DELETE"])
def api_delete_chunk(cid):
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(error="No session."), 400
    if sess.get("multifile"):
        return jsonify(error="Chunking is disabled for multi-file uploads."), 400

    keep, drop = [], []
    for c in sess["chunks"]:
        (drop if c["id"] == cid else keep).append(c)
    for c in drop:
        try:
            os.remove(c["path"])
        except OSError:
            pass
    sess["chunks"] = keep
    return jsonify(chunks=[{k: v for k, v in c.items() if k != "path"} for c in sess["chunks"]])


@app.route("/api/chunk/<int:cid>/download")
def download_chunk(cid):
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        abort(404)

    chunk = next((c for c in sess.get("chunks", []) if c["id"] == cid), None)
    if not chunk or not os.path.exists(chunk["path"]):
        abort(404)

    return send_file(
        chunk["path"],
        as_attachment=True,
        download_name=chunk["filename"]
    )


@app.route("/api/chunks/download")
def download_all_chunks():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess or not sess.get("chunks"):
        abort(404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for c in sess["chunks"]:
            if os.path.exists(c["path"]):
                z.write(c["path"], arcname=f"chunk_{c['id']}{os.path.splitext(c['filename'])[1]}")

    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="chunks.zip",
        mimetype="application/zip"
    )


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(error="Load a media file first."), 400

    d = request.get_json(force=True, silent=True) or {}
    engine = d.get("engine")
    language = d.get("language")
    device = d.get("device")
    model_size = d.get("model_size")
    diarize = bool(d.get("diarize"))
    diarize_method = d.get("diarize_method")           # 'pyannote' | 'ai_api'
    ai_provider = d.get("ai_provider")                 # 'gemini' | 'claude'
    api_key = d.get("api_key")
    num_speakers = d.get("num_speakers")

    with sess["lock"]:
        if not sess.get("source"):
            return jsonify(error="Load a media file first."), 400
        if sess["transcription_status"]["state"] == "running":
            return jsonify(error="A transcription is already running for this session."), 409
        source_files = list(sess["source_files"])
        is_multifile = sess.get("multifile")
        sess["transcriptions"] = []
        sess["transcript"] = None
        sess["transcription_status"] = {
            "state": "running", "completed": 0,
            "total": len(source_files) if is_multifile else 1, "error": None,
        }

    # ---- multi-file path: one transcription per source file, NOT concatenated ----
    if is_multifile:
        try:
            results = run_multifile_transcriptions(
                sess, source_files, engine, language, device, model_size,
                diarize, diarize_method, ai_provider, api_key, num_speakers,
            )
        except RuntimeError as e:
            with sess["lock"]:
                sess["transcription_status"].update(state="failed", error=str(e))
            return jsonify(error=str(e)), 500
        except Exception as e:
            with sess["lock"]:
                sess["transcription_status"].update(
                    state="failed", error=f"Transcription failed: {e}"
                )
            return jsonify(error=f"Transcription failed: {e}"), 500

        with sess["lock"]:
            sess["segments"] = []  # no combined segments in multi-file mode
            sess["transcription_status"]["state"] = "complete"

        return jsonify(
            multifile=True,
            transcriptions=results,
            # Provide an empty combined transcript so the single-textarea UI
            # still works if someone falls back to it; but frontend will use
            # `transcriptions` for multi-file.
            transcript="",
            segments=[],
        )

    # ---- single-file path (unchanged behavior, with CPU-parallel chunks) ----
    sources = sess["chunks"] if sess["chunks"] else [{
        "start": 0.0, "path": sess["source"]["path"],
    }]
    sources = sorted(sources, key=lambda c: c.get("start", 0.0))

    def _transcribe_item(item):
        if engine == "mlx" and diarize:
            segs = transcribe_file(
                item["path"], engine, language, device, model_size,
                hf_token=api_key, diarize=True
            )
        else:
            segs = transcribe_file(
                item["path"], engine, language, device, model_size
            )
        offset = item.get("start", 0.0)
        out = []
        for s in segs:
            segment = {
                "start": s["start"] + offset,
                "end": s["end"] + offset,
                "text": s["text"],
            }
            if s.get("speaker"):
                segment["speaker"] = s["speaker"]
            out.append(segment)
        return out

    try:
        if uses_gpu_device(engine, device) or force_sequential() or len(sources) <= 1:
            piece_lists = [_transcribe_item(item) for item in sources]
        else:
            piece_lists = run_cpu_jobs(
                [lambda item=item: _transcribe_item(item) for item in sources]
            )
        all_segments = []
        for pieces in piece_lists:
            all_segments.extend(pieces)
    except RuntimeError as e:
        with sess["lock"]:
            sess["transcription_status"].update(state="failed", error=str(e))
        return jsonify(error=str(e)), 500
    except Exception as e:
        with sess["lock"]:
            sess["transcription_status"].update(
                state="failed", error=f"Transcription failed: {e}"
            )
        return jsonify(error=f"Transcription failed: {e}"), 500

    # Handle diarization for non-MLX engines or when MLX didn't provide speakers
    if diarize and not any(s.get("speaker") for s in all_segments):
        try:
            if diarize_method == "pyannote":
                def _diarize_item(item):
                    t = run_pyannote(item["path"], api_key, num_speakers)
                    off = item.get("start", 0.0)
                    return [
                        {"start": x["start"] + off, "end": x["end"] + off, "speaker": x["speaker"]}
                        for x in t
                    ]

                if force_sequential() or len(sources) <= 1:
                    turn_lists = [_diarize_item(item) for item in sources]
                else:
                    # pyannote is GPU-heavy on many setups — serialize by default,
                    # but allow CPU-side pipeline overlap via DIARIX_CPU_WORKERS
                    # only when device is cpu; otherwise sequential.
                    if uses_gpu_device(engine, device) or (device or "").lower() != "cpu":
                        turn_lists = [_diarize_item(item) for item in sources]
                    else:
                        turn_lists = run_cpu_jobs(
                            [lambda item=item: _diarize_item(item) for item in sources]
                        )
                turns = []
                for tl in turn_lists:
                    turns.extend(tl)
                assign_speakers_from_turns(all_segments, turns)
            elif diarize_method == "ai_api":
                labels = run_ai_diarization(all_segments, ai_provider, api_key, num_speakers)
                for seg, label in zip(all_segments, labels):
                    seg["speaker"] = label
            else:
                with sess["lock"]:
                    sess["transcription_status"].update(
                        state="failed", error="Choose a diarization method (pyannote or AI API)."
                    )
                return jsonify(error="Choose a diarization method (pyannote or AI API)."), 400

        except RuntimeError as e:
            app.logger.exception("Diarization failed")
            with sess["lock"]:
                sess["transcription_status"].update(state="failed", error=str(e))
            return jsonify(error=str(e)), 500
        except Exception as e:
            app.logger.exception("Diarization failed")
            with sess["lock"]:
                sess["transcription_status"].update(
                    state="failed", error=f"Diarization failed: {e}"
                )
            return jsonify(error=f"Diarization failed: {e}"), 500

    transcript = segments_to_transcript(all_segments)
    sess["transcript"] = transcript
    sess["segments"] = all_segments
    sess["transcriptions"] = []
    with sess["lock"]:
        sess["transcription_status"].update(state="complete", completed=1)
    return jsonify(transcript=transcript, segments=all_segments)


@app.route("/api/transcribe/status")
def api_transcribe_status():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        return jsonify(error="No transcription session found."), 404
    with sess["lock"]:
        transcriptions = list(sess.get("transcriptions") or [])
        status = dict(sess["transcription_status"])
        if status["state"] == "running":
            status["completed"] = len(transcriptions)
    return jsonify(
        status=status,
        transcriptions=transcriptions,
    )


@app.route("/api/rename_speakers", methods=["POST"])
def api_rename_speakers():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess or not sess.get("transcript"):
        return jsonify(error="No transcript found."), 400

    data = request.get_json(force=True, silent=True) or {}
    speaker_mapping = data.get("speaker_mapping", {})

    if not speaker_mapping:
        return jsonify(error="No speaker mapping provided."), 400

    updated_segments = []
    for seg in sess["segments"]:
        new_seg = seg.copy()
        if seg.get("speaker") and seg["speaker"] in speaker_mapping:
            new_seg["speaker"] = speaker_mapping[seg["speaker"]]
        updated_segments.append(new_seg)

    lines = []
    for seg in updated_segments:
        prefix = f"{seg['speaker']}: " if seg.get("speaker") else ""
        lines.append(f"[{fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}] {prefix}{seg['text']}")
    transcript = "\n".join(lines)

    sess["segments"] = updated_segments
    sess["transcript"] = transcript

    return jsonify(transcript=transcript, segments=updated_segments)


@app.route("/api/download/transcription")
def download_transcription():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess or not sess.get("transcript"):
        abort(404)
    buf = io.BytesIO(sess["transcript"].encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="transcription.txt", mimetype="text/plain")


@app.route("/api/download/transcription/<int:idx>")
def download_transcription_indexed(idx):
    """Download a single per-file transcription (multi-file mode)."""
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        abort(404)
    transcriptions = sess.get("transcriptions") or []
    if idx < 0 or idx >= len(transcriptions):
        abort(404)
    item = transcriptions[idx]
    base = os.path.splitext(item.get("name", f"transcription_{idx}"))[0] or f"transcription_{idx}"
    fname = f"{base}.txt"
    buf = io.BytesIO(item["transcript"].encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="text/plain")


@app.route("/api/download/transcription/file/<file_id>")
def download_transcription_file(file_id):
    """Download a per-file transcript by stable source id while results arrive."""
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess:
        abort(404)
    item = next(
        (result for result in sess.get("transcriptions", [])
         if result.get("id") == file_id),
        None,
    )
    if item is None:
        abort(404)
    base = os.path.splitext(item.get("name", "transcription"))[0] or "transcription"
    fname = f"{base}.txt"
    buf = io.BytesIO(item["transcript"].encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="text/plain")


@app.route("/api/download/zip")
def download_zip():
    sid = get_sid_or_none()
    sess = get_session(sid) if sid else None
    if not sess or not sess.get("source"):
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if sess.get("transcriptions"):
            # multi-file mode: one txt per source file
            for i, item in enumerate(sess["transcriptions"]):
                base = os.path.splitext(item.get("name", f"transcription_{i}"))[0] or f"transcription_{i}"
                z.writestr(f"transcriptions/{base}.txt", item["transcript"])
        elif sess.get("transcript"):
            z.writestr("transcription.txt", sess["transcript"])
        if sess["chunks"]:
            for c in sess["chunks"]:
                if os.path.exists(c["path"]):
                    z.write(c["path"], arcname=f"chunks/{c['filename']}")
        elif sess.get("multifile"):
            for f in sess["source_files"]:
                if os.path.exists(f["path"]):
                    z.write(f["path"], arcname=f"sources/{f['name']}")
        else:
            z.write(sess["source"]["path"], arcname=f"source{sess['source']['ext']}")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="media_editor_export.zip", mimetype="application/zip")


# --------------------------------------------------------------------------
# frontend
# --------------------------------------------------------------------------

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Media editor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0b0c0e; --panel:#16181b; --panel-2:#1c1f23; --border:#2b2f34;
  --text:#edebe6; --text-dim:#9296a0; --text-faint:#5c6068;
  --amber:#e8a33d; --amber-dim:#8a6428; --teal:#5eead4; --red:#e5484d;
  --radius:10px;
  --font-display:'Space Grotesk',sans-serif; --font-body:'Inter',sans-serif; --font-mono:'IBM Plex Mono',monospace;
}
*{box-sizing:border-box;}
html,body{height:100%;}
body{
  margin:0; background:var(--bg); color:var(--text); font-family:var(--font-body);
  display:flex; flex-direction:column; overflow:hidden;
}
::selection{background:var(--amber); color:#161616;}
::-webkit-scrollbar{width:10px; height:10px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border); border-radius:6px;}
::-webkit-scrollbar-thumb:hover{background:var(--text-faint);}

header{
  flex:0 0 auto; padding:14px 22px; border-bottom:1px solid var(--border);
  position:relative; overflow:hidden;
  background:
    repeating-linear-gradient(115deg, rgba(232,163,61,0.05) 0 2px, transparent 2px 26px),
    var(--panel);
}
header h2{
  margin:0; font-family:var(--font-display); font-weight:600; font-size:17px;
  letter-spacing:0.02em; display:flex; align-items:center; gap:10px;
}
header h2::before{
  content:''; width:9px; height:9px; border-radius:50%; background:var(--amber);
  box-shadow:0 0 10px 1px var(--amber); flex:0 0 auto;
}

main{flex:1 1 auto; min-height:0; display:flex; padding:16px 22px 22px;}
.main-container{
  flex:1; min-height:0; display:flex; flex-direction:column;
  border:1px solid var(--border); border-radius:var(--radius); background:var(--panel);
  overflow:hidden;
}

.tabs{flex:1; min-height:0; display:flex; flex-direction:column;}
.tab-nav{
  flex:0 0 auto; display:flex; border-bottom:1px solid var(--border); background:var(--panel-2);
  overflow-x:auto;
}
.tab-nav button{
  font-family:var(--font-body); font-weight:600; font-size:13px; color:var(--text-dim);
  background:none; border:none; padding:13px 18px 11px; cursor:pointer; white-space:nowrap;
  border-bottom:2px solid transparent; display:flex; align-items:center; gap:8px;
  transition:color .15s;
}
.tab-nav button .eyebrow{
  font-family:var(--font-mono); font-size:10px; color:var(--text-faint); letter-spacing:.05em;
}
.tab-nav button:hover{color:var(--text);}
.tab-nav button.active{color:var(--text); border-bottom-color:var(--amber);}
.tab-nav button.active .eyebrow{color:var(--amber);}
.tab-nav button:disabled{color:var(--text-faint); cursor:not-allowed;}

.tab-panel{display:none; flex:1; min-height:0; overflow-y:auto; padding:20px 24px 28px;}
.tab-panel.active{display:flex; flex-direction:column; gap:18px;}

.field{display:flex; flex-direction:column; gap:6px;}
.field label{font-size:12px; color:var(--text-dim); font-weight:500;}
.row{display:flex; gap:14px; flex-wrap:wrap;}
.row > .field{flex:1; min-width:150px;}

select, input[type=text], input[type=url], input[type=number], input[type=password], textarea{
  background:var(--panel-2); border:1px solid var(--border); color:var(--text);
  border-radius:7px; padding:9px 11px; font-family:var(--font-body); font-size:13.5px;
  outline:none; transition:border-color .15s;
}
select:focus, input:focus, textarea:focus{border-color:var(--amber);}
select:disabled, input:disabled{opacity:.4; cursor:not-allowed;}

input[type=file]{color:var(--text-dim); font-size:13px;}

.btn{
  font-family:var(--font-body); font-weight:600; font-size:13.5px; cursor:pointer;
  border-radius:7px; padding:10px 18px; border:1px solid var(--amber); background:var(--amber);
  color:#181205; transition:filter .15s, transform .05s;
}
.btn:hover{filter:brightness(1.08);}
.btn:active{transform:translateY(1px);}
.btn:disabled{background:var(--panel-2); border-color:var(--border); color:var(--text-faint); cursor:not-allowed;}
.btn.secondary{background:transparent; color:var(--text); border-color:var(--border);}
.btn.secondary:hover{border-color:var(--text-dim); filter:none;}
.btn.small{padding:6px 11px; font-size:12px;}
.btn.danger{background:transparent; border-color:var(--red); color:var(--red);}

fieldset{border:1px solid var(--border); border-radius:9px; padding:14px 16px; margin:0; display:flex; flex-direction:column; gap:12px;}
fieldset[disabled]{opacity:.35;}
legend{font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--text-dim); padding:0 6px;}

.checkline{display:flex; align-items:center; gap:9px; font-size:13.5px;}
.checkline input[type=checkbox]{accent-color:var(--amber); width:15px; height:15px;}
.subrow{margin-left:26px; display:flex; flex-direction:column; gap:10px;}

.hidden{display:none !important;}

/* --- load tab --- */
.mode-toggle{display:flex; gap:0; width:fit-content; border:1px solid var(--border); border-radius:8px; overflow:hidden;}
.mode-toggle button{
  background:var(--panel-2); border:none; color:var(--text-dim); padding:8px 16px; font-size:13px;
  font-family:var(--font-body); font-weight:500; cursor:pointer;
}
.mode-toggle button.active{background:var(--amber); color:#181205; font-weight:600;}

/* --- media info panel --- */
.media-info-panel{
  display:flex; flex-direction:column; gap:8px; background:var(--panel-2);
  border:1px solid var(--border); border-radius:8px; padding:14px 16px;
  font-size:13px;
}
.media-info-panel .info-title{
  font-family:var(--font-display); font-weight:600; font-size:14px; color:var(--amber);
}
.media-info-panel .info-grid{
  display:grid; grid-template-columns:1fr 1fr; gap:8px;
}
.media-info-panel .info-item{
  display:flex; flex-direction:column; gap:3px;
}
.media-info-panel .info-label{
  font-size:11px; color:var(--text-faint); text-transform:uppercase; letter-spacing:.04em;
}
.media-info-panel .info-value{
  font-family:var(--font-mono); font-size:12.5px; color:var(--text);
}

/* --- format selection --- */
.format-selector{
  display:flex; flex-direction:column; gap:10px;
  background:var(--panel-2); border:1px solid var(--border); border-radius:8px; padding:14px 16px;
}
.format-selector .format-list{
  max-height:200px; overflow-y:auto; display:flex; flex-direction:column; gap:6px;
}
.format-selector .format-item{
  display:flex; align-items:center; gap:8px; padding:8px 10px;
  background:var(--panel); border:1px solid var(--border); border-radius:6px;
  cursor:pointer; transition:border-color .15s;
}
.format-selector .format-item:hover{border-color:var(--amber);}
.format-selector .format-item.selected{border-color:var(--amber); background:rgba(232,163,61,0.1);}
.format-selector .format-item input[type=radio]{accent-color:var(--amber);}
.format-selector .format-item .format-label{
  flex:1; font-family:var(--font-mono); font-size:12px; color:var(--text);
}
.format-selector .format-item .format-meta{
  font-size:11px; color:var(--text-faint);
}

/* --- upload file pool --- */
.file-pool{
  display:flex; flex-direction:column; gap:8px;
  background:var(--panel-2); border:1px solid var(--border); border-radius:8px; padding:12px 14px;
}
.file-pool .pool-title{
  font-family:var(--font-display); font-weight:600; font-size:13.5px; color:var(--amber);
}
.file-pool .pool-header{display:flex; align-items:center; justify-content:space-between; gap:10px;}
ul#filePoolList{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px;}
ul#filePoolList li{
  display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap;
  background:var(--panel); border:1px solid var(--border); border-radius:7px; padding:9px 12px;
  font-size:12.5px;
}
ul#filePoolList li .fname{
  font-family:var(--font-mono); color:var(--text); word-break:break-all; flex:1;
}
ul#filePoolList li .fmeta{color:var(--text-dim); font-family:var(--font-mono); font-size:11.5px; white-space:nowrap;}
ul#filePoolList li .pool-actions{display:flex; gap:6px; flex-wrap:wrap;}
ul#filePoolList li .pool-rename{display:flex; gap:6px; flex:1 1 220px; min-width:180px;}
ul#filePoolList li .pool-rename input{min-width:0; flex:1; padding:6px 8px; font-size:12px;}

/* --- preview & cut --- */
.preview-wrap{background:#000; border:1px solid var(--border); border-radius:9px; overflow:hidden; max-height:38vh; display:flex; align-items:center; justify-content:center;}
video, audio{width:100%; display:block;}
audio{padding:16px;}
.timeline{display:flex; flex-direction:column; gap:8px;}
.ruler{
  height:16px; border-radius:3px; position:relative;
  background:repeating-linear-gradient(90deg, var(--border) 0 1px, transparent 1px 20px);
}
.range-stack{position:relative; height:30px;}
.range-stack input[type=range]{
  position:absolute; left:0; right:0; top:0; width:100%; margin:0; height:30px;
  background:none; -webkit-appearance:none; pointer-events:none;
}
.range-stack input[type=range]::-webkit-slider-runnable-track{height:4px; background:var(--border); border-radius:2px;}
.range-stack input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none; pointer-events:auto; width:16px; height:16px; border-radius:50%;
  margin-top:-6px; cursor:grab; border:2px solid #0b0c0e;
}
#startRange::-webkit-slider-thumb{background:var(--teal);}
#endRange::-webkit-slider-thumb{background:var(--amber);}
.range-stack input[type=range]::-moz-range-thumb{
  pointer-events:auto; width:16px; height:16px; border-radius:50%; border:2px solid #0b0c0e; cursor:grab;
}
#startRange::-moz-range-thumb{background:var(--teal);}
#endRange::-moz-range-thumb{background:var(--amber);}

.timers{display:flex; align-items:flex-end; gap:14px; flex-wrap:wrap;}
.timer-box{display:flex; flex-direction:column; gap:5px;}
.timer-box .tag{font-family:var(--font-mono); font-size:10.5px; letter-spacing:.05em;}
.timer-box.start .tag{color:var(--teal);}
.timer-box.end .tag{color:var(--amber);}
.timer-box input{font-family:var(--font-mono); width:110px;}
.timer-actions{display:flex; gap:6px;}

ul#chunkList{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px;}
ul#chunkList li{
  display:flex; align-items:center; justify-content:space-between; gap:10px;
  background:var(--panel-2); border:1px solid var(--border); border-radius:7px; padding:9px 12px;
  font-family:var(--font-mono); font-size:12.5px;
}
ul#chunkList li .meta{color:var(--text-dim);}
ul#chunkList li .idx{color:var(--amber); font-weight:600;}
.empty-note{color:var(--text-faint); font-size:12.5px; font-style:italic;}

/* --- transcription tab --- */
.speaker-name-list{display:flex; flex-direction:column; gap:8px;}
.speaker-name-list .sn-row{display:flex; gap:8px; align-items:center;}
.speaker-name-list .sn-row span{font-family:var(--font-mono); color:var(--text-faint); font-size:11.5px; width:64px;}

/* --- extract tab --- */
textarea#transcriptOut{flex:1; min-height:220px; resize:vertical; font-family:var(--font-mono); font-size:12.5px; line-height:1.6;}
.download-row{display:flex; gap:10px; flex-wrap:wrap;}

/* --- multi-file transcription results --- */
.multi-results{display:flex; flex-direction:column; gap:16px;}
.multi-result-item{
  display:flex; flex-direction:column; gap:8px;
  background:var(--panel-2); border:1px solid var(--border); border-radius:9px; padding:12px 14px;
}
.multi-result-item .mr-header{
  display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap;
}
.multi-result-item .mr-name{
  font-family:var(--font-display); font-weight:600; font-size:13.5px; color:var(--amber);
  word-break:break-all;
}
.multi-result-item textarea{
  min-height:160px; resize:vertical; font-family:var(--font-mono); font-size:12.5px; line-height:1.6;
}

.toast{
  position:fixed; bottom:18px; left:50%; transform:translateX(-50%) translateY(10px);
  background:#20120f; border:1px solid var(--red); color:#ffd8d6; padding:11px 18px;
  border-radius:8px; font-size:13px; max-width:min(560px,90vw); opacity:0; pointer-events:none;
  transition:opacity .2s, transform .2s; z-index:50;
}
.toast.show{opacity:1; transform:translateX(-50%) translateY(0);}

.spinner{
  width:14px; height:14px; border-radius:50%; border:2px solid rgba(0,0,0,.25);
  border-top-color:#181205; animation:spin .7s linear infinite; display:inline-block; vertical-align:-2px;
}
@keyframes spin{to{transform:rotate(360deg);}}

footer{flex:0 0 auto; padding:9px 22px; text-align:center; font-size:11px; color:var(--text-faint); font-family:var(--font-mono);}

@media (max-width:720px){
  main{padding:10px;}
  header{padding:12px 16px;}
  .tab-nav button{padding:11px 12px 9px; font-size:12px;}
  .tab-panel{padding:16px;}
  .row > .field{min-width:120px;}
  .preview-wrap{max-height:26vh;}
  .media-info-panel .info-grid{grid-template-columns:1fr;}
}
</style>
</head>
<body>

<header><h2>Media editor</h2></header>

<main>
  <div class="main-container">
    <div class="tabs">
      <div class="tab-nav">
        <button class="tab-btn active" data-tab="load"><span class="eyebrow">01</span>Load media</button>
        <button class="tab-btn" data-tab="cut" disabled><span class="eyebrow">02</span>Preview &amp; cut</button>
        <button class="tab-btn" data-tab="transcribe" disabled><span class="eyebrow">03</span>Transcription</button>
        <button class="tab-btn" data-tab="extract" disabled><span class="eyebrow">04</span>Results</button>
      </div>

      <!-- TAB 1: load media -->
      <div class="tab-panel active" data-panel="load">
        <div class="field">
          <label>Source</label>
          <div class="mode-toggle">
            <button type="button" id="modeUpload" class="active">Upload file(s)</button>
            <button type="button" id="modeUrl">Enter URL</button>
          </div>
        </div>
        <div class="field" id="uploadField">
          <label for="fileInput">Media file(s) — add one or more files at a time; the upload pool stays open for more files</label>
          <input type="file" id="fileInput" accept="video/*,audio/*" multiple>
        </div>
        <div class="field hidden" id="urlField">
          <label for="urlInput">Media URL</label>
          <input type="url" id="urlInput" placeholder="https://example.com/clip.mp4">
        </div>

        <!-- YouTube format selector (only shown for YouTube URLs) -->
        <div class="field hidden" id="youtubeFormatField">
          <label>Download format</label>
          <button class="btn secondary small" id="fetchFormatsBtn" type="button">Fetch available formats</button>
          <div id="formatSelector" class="format-selector hidden">
            <div class="format-list" id="formatList"></div>
          </div>
        </div>

        <div><button class="btn" id="loadBtn">Add files</button></div>

        <!-- Upload pool: add / reorder / delete before transcription -->
        <div id="filePool" class="file-pool hidden">
          <div class="pool-header">
            <div class="pool-title">Audio / video pool <span id="filePoolCount" style="color:var(--text-dim); font-weight:500; font-size:12px;"></span></div>
            <button class="btn danger small" id="clearFilesBtn" type="button">Wipe list</button>
          </div>
          <div class="empty-note" id="filePoolEmpty">No files yet — choose one or more files above.</div>
          <ul id="filePoolList"></ul>
          <div class="empty-note">Transcription order follows this list. Rename or remove files, or use ↑ / ↓ to reorder.</div>
        </div>

        <!-- Media info panel -->
        <div id="mediaInfoPanel" class="media-info-panel hidden">
          <div class="info-title">Media Information</div>
          <div class="info-grid" id="mediaInfoGrid"></div>
        </div>
      </div>

      <!-- TAB 2: preview and cut -->
      <div class="tab-panel" data-panel="cut">
        <div id="multiFileNotice" class="empty-note hidden"></div>
        <div class="preview-wrap" id="previewWrap">
          <video id="videoPreview" controls></video>
        </div>
        <div class="timeline">
          <div class="ruler"></div>
          <div class="range-stack">
            <input type="range" id="startRange" min="0" max="100" step="0.01" value="0">
            <input type="range" id="endRange" min="0" max="100" step="0.01" value="100">
          </div>
        </div>
        <div class="timers">
          <div class="timer-box start">
            <span class="tag">IN</span>
            <input type="text" id="startTimer" value="00:00.00">
            <div class="timer-actions"><button class="btn secondary small" id="setStartBtn">Use playhead</button></div>
          </div>
          <div class="timer-box end">
            <span class="tag">OUT</span>
            <input type="text" id="endTimer" value="00:00.00">
            <div class="timer-actions"><button class="btn secondary small" id="setEndBtn">Use playhead</button></div>
          </div>
          <button class="btn" id="addChunkBtn">Add chunk</button>
        </div>
        <div class="field">
          <label>Chunks</label>
          <ul id="chunkList"></ul>
          <div class="empty-note" id="chunkEmptyNote">No chunks yet — transcription will use the full file.</div>
          <div style="margin-top: 10px;">
            <button class="btn secondary small hidden" id="downloadAllChunksBtn">Download all chunks</button>
          </div>
        </div>
      </div>

      <!-- TAB 3: transcription controls -->
      <div class="tab-panel" data-panel="transcribe">
        <div class="row">
          <div class="field">
            <label for="engineSel">Engine</label>
            <select id="engineSel">
              <option value="openai-whisper">openai-whisper</option>
              <option value="faster">faster-whisper</option>
              <option value="mlx">whispermlx (Apple Silicon)</option>
            </select>
          </div>
          <div class="field">
            <label for="langSel">Language</label>
            <select id="langSel">
              <option value="auto">Auto detect</option>
              <option value="de">German</option>
              <option value="en" selected>English</option>
              <option value="fr">French</option>
              <option value="ro">Romanian</option>
            </select>
          </div>
          <div class="field">
            <label for="deviceSel">Device</label>
            <select id="deviceSel">
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
              <option value="metal">Metal</option>
            </select>
          </div>
          <div class="field">
            <label for="modelSel">Model size</label>
            <select id="modelSel">
              <option value="tiny">tiny</option>
              <option value="base">base</option>
              <option value="small" selected>small</option>
              <option value="medium">medium</option>
              <option value="large-v3">large-v3</option>
            </select>
          </div>
        </div>

        <div class="checkline">
          <input type="checkbox" id="diarizeCk"><label for="diarizeCk">Identify speakers (diarization)</label>
        </div>

        <fieldset id="diarizeFields" disabled>
          <legend>Speaker identification</legend>

          <div class="checkline">
            <input type="checkbox" id="pyannoteCk"><label for="pyannoteCk">pyannote.audio (audio-based, needs a Hugging Face token)</label>
          </div>
          <div class="checkline">
            <input type="checkbox" id="aiApiCk"><label for="aiApiCk">AI API (text-based heuristic labeling)</label>
          </div>
          <div class="subrow hidden" id="aiApiSub">
            <div class="field" style="max-width:220px">
              <label for="aiProviderSel">Provider</label>
              <select id="aiProviderSel">
                <option value="gemini">Gemini</option>
                <option value="claude">Claude</option>
              </select>
            </div>
          </div>

          <div class="checkline">
            <input type="checkbox" id="apiKeyCk" disabled><label for="apiKeyCk">API key</label>
          </div>
          <div class="subrow hidden" id="apiKeySub">
            <input type="password" id="apiKeyInput" placeholder="Paste key or token">
          </div>
        </fieldset>

        <div><button class="btn" id="transcribeBtn">Transcribe</button></div>
      </div>

      <!-- TAB 4: extract results -->
      <div class="tab-panel" data-panel="extract" style="flex:1;">
        <!-- Single-file mode UI -->
        <div id="singleResultWrap" style="display:flex; flex-direction:column; gap:18px; flex:1;">
          <textarea id="transcriptOut" placeholder="Your transcription will appear here." readonly></textarea>
          <div class="download-row">
            <button class="btn secondary" id="downloadTxtBtn">Download transcription</button>
            <button class="btn secondary" id="downloadZipBtn">Download zip file</button>
          </div>
          <div id="speakerRenamePanel" class="hidden" style="margin-top: 10px;">
            <fieldset>
              <legend>Rename speakers</legend>
              <div id="speakerRenameList"></div>
              <div style="margin-top: 10px;">
                <button class="btn" id="applySpeakerNamesBtn">Apply names</button>
              </div>
            </fieldset>
          </div>
        </div>

        <!-- Multi-file mode UI: one textarea + download button per file -->
        <div id="multiResultWrap" class="hidden" style="display:flex; flex-direction:column; gap:18px; flex:1;">
          <div class="download-row">
            <button class="btn secondary" id="downloadZipBtnMulti">Download zip file</button>
          </div>
          <div class="empty-note" id="multiResultStatus"></div>
          <div id="multiResultList" class="multi-results"></div>
        </div>
      </div>

    </div>
  </div>
</main>

<footer>
  <div class="copyright-notice">© 2026 Media Editor — local ffmpeg &amp; Whisper processing</div>
</footer>

<div class="toast" id="toast"></div>

<script>
const state = {
  kind:null, duration:0, chunks:[], segments:[],
  selectedFormat:null, youtubeFormats:[],
  multifile:false, fileCount:0, files:[],
  transcriptions:[],
};

// ---------- utils ----------
function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast._h);
  toast._h = setTimeout(()=>t.classList.remove('show'), 120000);
}
function fmtTs(t){
  t = Math.max(0,t);
  const m = Math.floor(t/60);
  const s = (t - m*60).toFixed(2).padStart(5,'0');
  return `${String(m).padStart(2,'0')}:${s}`;
}
function parseTs(str){
  const m = str.trim().match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if(!m) return null;
  return parseInt(m[1],10)*60 + parseFloat(m[2]);
}
function formatBytes(bytes){
  if (!bytes) return 'N/A';
  const units = ['B', 'KB', 'MB', 'GB'];
  let unitIndex = 0;
  let size = bytes;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}
async function api(url, opts){
  const res = await fetch(url, opts);
  let data = null;
  try{ data = await res.json(); }catch(e){}
  if(!res.ok){
    throw new Error((data && data.error) || `Request failed (${res.status})`);
  }
  return data;
}

// ---------- tabs ----------
const tabButtons = document.querySelectorAll('.tab-btn');
tabButtons.forEach(btn=>{
  btn.addEventListener('click', ()=>{
    if(btn.disabled) return;
    tabButtons.forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.querySelector(`.tab-panel[data-panel="${btn.dataset.tab}"]`).classList.add('active');
  });
});
function goTab(name){ document.querySelector(`.tab-btn[data-tab="${name}"]`).click(); }
function unlockTabs(){ tabButtons.forEach(b=>{ if(b.dataset.tab!=='load') b.disabled=false; }); }

// ---------- tab 1: load ----------
const modeUpload = document.getElementById('modeUpload');
const modeUrl = document.getElementById('modeUrl');
const uploadField = document.getElementById('uploadField');
const urlField = document.getElementById('urlField');
const youtubeFormatField = document.getElementById('youtubeFormatField');
const formatSelector = document.getElementById('formatSelector');
const formatList = document.getElementById('formatList');
const fetchFormatsBtn = document.getElementById('fetchFormatsBtn');
const mediaInfoPanel = document.getElementById('mediaInfoPanel');
const mediaInfoGrid = document.getElementById('mediaInfoGrid');

modeUpload.addEventListener('click', ()=>{
  modeUpload.classList.add('active'); modeUrl.classList.remove('active');
  uploadField.classList.remove('hidden'); urlField.classList.add('hidden');
  youtubeFormatField.classList.add('hidden');
  document.getElementById('loadBtn').textContent = 'Add files';
});
modeUrl.addEventListener('click', ()=>{
  modeUrl.classList.add('active'); modeUpload.classList.remove('active');
  urlField.classList.remove('hidden'); uploadField.classList.add('hidden');
  document.getElementById('loadBtn').textContent = 'Load from URL';
});

document.getElementById('urlInput').addEventListener('input', ()=>{
  const url = document.getElementById('urlInput').value.trim();
  const isYoutube = url.includes('youtube.com') || url.includes('youtu.be');

  if (modeUrl.classList.contains('active') && isYoutube) {
    youtubeFormatField.classList.remove('hidden');
    formatSelector.classList.add('hidden');
    state.selectedFormat = null;
    state.youtubeFormats = [];
  } else {
    youtubeFormatField.classList.add('hidden');
  }
});

fetchFormatsBtn.addEventListener('click', async ()=>{
  const url = document.getElementById('urlInput').value.trim();
  if (!url) {
    toast('Enter a YouTube URL first.');
    return;
  }

  fetchFormatsBtn.disabled = true;
  fetchFormatsBtn.innerHTML = '<span class="spinner"></span> Fetching…';

  try {
    const data = await api('/api/youtube/formats', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url}),
    });

    state.youtubeFormats = data.formats;
    renderFormatList(data.formats);
    formatSelector.classList.remove('hidden');
  } catch(e) {
    toast(e.message);
  } finally {
    fetchFormatsBtn.disabled = false;
    fetchFormatsBtn.textContent = 'Fetch available formats';
  }
});

function renderFormatList(formats){
  formatList.innerHTML = '';

  if (!formats || formats.length === 0) {
    formatList.innerHTML = '<div class="empty-note">No formats available.</div>';
    return;
  }

  const filtered = formats.filter(f => {
    return (f.vcodec !== 'none' || f.acodec !== 'none') &&
           (f.ext === 'mp4' || f.ext === 'webm' || f.ext === 'm4a' || f.ext === 'opus');
  });

  const sorted = filtered.sort((a, b) => {
    const qualityRank = {best: 0, high: 1, medium: 2, low: 3, worst: 4};
    return (qualityRank[a.quality] || 5) - (qualityRank[b.quality] || 5);
  });

  sorted.slice(0, 20).forEach(f => {
    const label = document.createElement('label');
    label.className = 'format-item';

    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'ytFormat';
    radio.value = f.format_id;
    radio.addEventListener('change', ()=>{
      state.selectedFormat = f.format_id;
      document.querySelectorAll('.format-item').forEach(item => item.classList.remove('selected'));
      label.classList.add('selected');
    });

    const formatLabel = document.createElement('span');
    formatLabel.className = 'format-label';
    const sizeInfo = f.filesize ? formatBytes(f.filesize) : (f.filesize_approx ? `~${formatBytes(f.filesize_approx)}` : 'N/A');
    formatLabel.textContent = `${f.ext.toUpperCase()} - ${f.resolution}${f.fps ? ` @ ${f.fps}fps` : ''} - ${sizeInfo}`;

    const formatMeta = document.createElement('span');
    formatMeta.className = 'format-meta';
    const parts = [];
    if (f.vcodec !== 'none') parts.push(`video: ${f.vcodec}`);
    if (f.acodec !== 'none') parts.push(`audio: ${f.acodec}`);
    if (f.format_note) parts.push(f.format_note);
    formatMeta.textContent = parts.join(' | ');

    label.appendChild(radio);
    label.appendChild(formatLabel);
    label.appendChild(formatMeta);
    formatList.appendChild(label);
  });
}

function displayMediaInfo(data){
  mediaInfoPanel.classList.remove('hidden');
  mediaInfoGrid.innerHTML = '';

  const items = [];

  if (data.file_count > 1) {
    items.push({label: 'Files', value: String(data.file_count)});
  }
  items.push({label: 'Type', value: data.kind === 'video' ? 'Video' : 'Audio'});
  items.push({label: 'Duration', value: fmtTs(data.duration)});
  items.push({label: 'File size', value: formatBytes(data.file_size)});
  items.push({label: 'Format', value: data.format || 'N/A'});

  if (data.video_info) {
    items.push({label: 'Video codec', value: data.video_info.codec || 'N/A'});
    items.push({label: 'Resolution', value: data.video_info.width ? `${data.video_info.width}x${data.video_info.height}` : 'N/A'});
    items.push({label: 'Frame rate', value: data.video_info.fps || 'N/A'});
    items.push({label: 'Video bitrate', value: data.video_info.bitrate ? formatBytes(data.video_info.bitrate) + '/s' : 'N/A'});
  }

  if (data.audio_info) {
    items.push({label: 'Audio codec', value: data.audio_info.codec || 'N/A'});
    items.push({label: 'Sample rate', value: data.audio_info.sample_rate ? `${data.audio_info.sample_rate} Hz` : 'N/A'});
    items.push({label: 'Channels', value: data.audio_info.channels || 'N/A'});
    items.push({label: 'Audio bitrate', value: data.audio_info.bitrate ? formatBytes(data.audio_info.bitrate) + '/s' : 'N/A'});
  }

  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'info-item';
    div.innerHTML = `
      <span class="info-label">${item.label}</span>
      <span class="info-value">${item.value}</span>
    `;
    mediaInfoGrid.appendChild(div);
  });
}

function applyLoadedFiles(data){
  state.kind = data.kind;
  state.duration = data.duration || 0;
  state.chunks = [];
  state.multifile = !!data.multifile;
  state.fileCount = data.file_count || 0;
  state.files = data.files || [];
  state.transcriptions = [];

  renderFilePool();

  if (!state.files.length) {
    mediaInfoPanel.classList.add('hidden');
    document.querySelector('.tab-btn[data-tab="cut"]').disabled = true;
    document.querySelector('.tab-btn[data-tab="transcribe"]').disabled = true;
    document.querySelector('.tab-btn[data-tab="extract"]').disabled = true;
    clearResultsUI();
    return;
  }

  if (data.src) setupPreviewElement(data.kind, data.src);
  resetCutUI();
  renderChunkList();
  displayMediaInfo(data);
  applyMultifileMode(state.multifile);
  clearResultsUI();
  unlockTabs();
}

function renderFilePool(){
  const pool = document.getElementById('filePool');
  const list = document.getElementById('filePoolList');
  const empty = document.getElementById('filePoolEmpty');
  const count = document.getElementById('filePoolCount');
  const files = state.files || [];

  pool.classList.toggle('hidden', files.length === 0 && modeUrl.classList.contains('active'));
  if (modeUpload.classList.contains('active')) pool.classList.remove('hidden');

  count.textContent = files.length ? `(${files.length})` : '';
  empty.classList.toggle('hidden', files.length > 0);
  list.innerHTML = '';

  files.forEach((f, idx)=>{
    const li = document.createElement('li');
    li.dataset.id = f.id;

    const left = document.createElement('div');
    left.style.cssText = 'display:flex; flex-direction:column; gap:2px; flex:1; min-width:0;';
    const fileName = document.createElement('span');
    fileName.className = 'fname';
    fileName.textContent = `${idx+1}. ${f.name}`;
    const fileMeta = document.createElement('span');
    fileMeta.className = 'fmeta';
    fileMeta.textContent = `${f.kind || ''} · ${fmtTs(f.duration || 0)} · ${formatBytes(f.file_size || 0)}`;
    left.appendChild(fileName);
    left.appendChild(fileMeta);

    const rename = document.createElement('form');
    rename.className = 'pool-rename';
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.value = f.name;
    nameInput.setAttribute('aria-label', `New name for ${f.name}`);
    nameInput.maxLength = 255;
    const renameBtn = document.createElement('button');
    renameBtn.className = 'btn secondary small';
    renameBtn.type = 'submit';
    renameBtn.textContent = 'Rename';
    rename.addEventListener('submit', event => {
      event.preventDefault();
      renameFile(f.id, nameInput.value);
    });
    rename.appendChild(nameInput);
    rename.appendChild(renameBtn);

    const actions = document.createElement('div');
    actions.className = 'pool-actions';

    const upBtn = document.createElement('button');
    upBtn.className = 'btn secondary small';
    upBtn.textContent = '↑';
    upBtn.title = 'Move up';
    upBtn.disabled = idx === 0;
    upBtn.addEventListener('click', ()=>reorderFile(idx, idx-1));

    const downBtn = document.createElement('button');
    downBtn.className = 'btn secondary small';
    downBtn.textContent = '↓';
    downBtn.title = 'Move down';
    downBtn.disabled = idx === files.length - 1;
    downBtn.addEventListener('click', ()=>reorderFile(idx, idx+1));

    const delBtn = document.createElement('button');
    delBtn.className = 'btn danger small';
    delBtn.textContent = 'Remove';
    delBtn.addEventListener('click', ()=>deleteFile(f.id));

    actions.appendChild(upBtn);
    actions.appendChild(downBtn);
    actions.appendChild(delBtn);
    li.appendChild(left);
    li.appendChild(rename);
    li.appendChild(actions);
    list.appendChild(li);
  });
}

async function renameFile(fileId, name){
  try{
    const data = await api(`/api/files/${encodeURIComponent(fileId)}/rename`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name}),
    });
    applyLoadedFiles(data);
  }catch(e){ toast(e.message); }
}

async function reorderFile(fromIdx, toIdx){
  const files = state.files.slice();
  if (toIdx < 0 || toIdx >= files.length) return;
  const [item] = files.splice(fromIdx, 1);
  files.splice(toIdx, 0, item);
  const order = files.map(f => f.id);
  try{
    const data = await api('/api/files/reorder', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({order}),
    });
    applyLoadedFiles(data);
  }catch(e){ toast(e.message); }
}

async function deleteFile(fileId){
  try{
    const data = await api(`/api/files/${encodeURIComponent(fileId)}`, {method:'DELETE'});
    applyLoadedFiles(data);
  }catch(e){ toast(e.message); }
}

document.getElementById('clearFilesBtn').addEventListener('click', async ()=>{
  if (!state.files.length || !window.confirm('Wipe all files from the upload list?')) return;
  try{
    const data = await api('/api/files', {method:'DELETE'});
    applyLoadedFiles(data);
    goTab('load');
  }catch(e){ toast(e.message); }
});

async function resetWorkspace(){
  try{
    await api('/api/session/reset', {method:'POST'});
    state.kind = null;
    state.duration = 0;
    state.chunks = [];
    state.segments = [];
    state.multifile = false;
    state.fileCount = 0;
    state.files = [];
    state.transcriptions = [];
    renderFilePool();
    mediaInfoPanel.classList.add('hidden');
    clearResultsUI();
  }catch(e){
    toast('Could not reset workspace: ' + e.message);
  }
}

const previewWrap = document.getElementById('previewWrap');
let mediaEl = document.getElementById('videoPreview');

document.getElementById('loadBtn').addEventListener('click', async ()=>{
  const btn = document.getElementById('loadBtn');
  const fd = new FormData();

  if(modeUpload.classList.contains('active')){
    const files = document.getElementById('fileInput').files;
    if(!files || files.length === 0){ toast('Choose at least one file.'); return; }
    for (const f of files) fd.append('files', f);
  } else {
    const u = document.getElementById('urlInput').value.trim();
    if(!u){ toast('Enter a URL first.'); return; }
    fd.append('url', u);

    if (state.selectedFormat && (u.includes('youtube.com') || u.includes('youtu.be'))) {
      fd.append('format_option', state.selectedFormat);
    }
  }

  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Loading…';
  try{
    const data = await api('/api/load', {method:'POST', body: fd});
    state.selectedFormat = null;
    applyLoadedFiles(data);
    document.getElementById('fileInput').value = '';
    if (state.multifile) {
      goTab('transcribe');
    } else if (state.files.length) {
      goTab('cut');
    }
  }catch(e){ toast(e.message); }
  finally{ btn.disabled = false; btn.textContent = modeUpload.classList.contains('active') ? 'Add files' : 'Load from URL'; }
});

// Wipe leftover audio from any previous browser/server session on first paint.
resetWorkspace();

function applyMultifileMode(isMulti){
  const cutBtn = document.querySelector('.tab-btn[data-tab="cut"]');
  cutBtn.disabled = isMulti;

  document.getElementById('addChunkBtn').disabled = isMulti;
  document.getElementById('startRange').disabled = isMulti;
  document.getElementById('endRange').disabled = isMulti;
  document.getElementById('setStartBtn').disabled = isMulti;
  document.getElementById('setEndBtn').disabled = isMulti;

  const note = document.getElementById('chunkEmptyNote');
  const notice = document.getElementById('multiFileNotice');
  if (isMulti) {
    note.textContent = 'Multi-file mode: chunking is disabled. All files will be transcribed individually.';
    note.classList.remove('hidden');
    notice.textContent = 'Multi-file upload: the Preview & cut tab is unavailable.';
    notice.classList.remove('hidden');
  } else {
    note.textContent = 'No chunks yet — transcription will use the full file.';
    notice.classList.add('hidden');
  }
}

function setupPreviewElement(kind, src){
  previewWrap.innerHTML = '';
  if(kind === 'video'){
    mediaEl = document.createElement('video');
    mediaEl.id = 'videoPreview'; mediaEl.controls = true;
  } else {
    mediaEl = document.createElement('audio');
    mediaEl.id = 'videoPreview'; mediaEl.controls = true;
  }
  mediaEl.src = src;
  previewWrap.appendChild(mediaEl);
}

// ---------- tab 2: preview & cut ----------
const startRange = document.getElementById('startRange');
const endRange = document.getElementById('endRange');
const startTimer = document.getElementById('startTimer');
const endTimer = document.getElementById('endTimer');

function resetCutUI(){
  startRange.min = 0; startRange.max = state.duration; startRange.step = 0.01; startRange.value = 0;
  endRange.min = 0; endRange.max = state.duration; endRange.step = 0.01; endRange.value = state.duration;
  startTimer.value = fmtTs(0);
  endTimer.value = fmtTs(state.duration);
}

function syncFromRanges(){
  let s = parseFloat(startRange.value), e = parseFloat(endRange.value);
  if(s > e - 0.05){ s = Math.max(0, e - 0.05); startRange.value = s; }
  startTimer.value = fmtTs(s);
  endTimer.value = fmtTs(e);
}
startRange.addEventListener('input', ()=>{ syncFromRanges(); if(mediaEl) mediaEl.currentTime = parseFloat(startRange.value); });
endRange.addEventListener('input', ()=>{ syncFromRanges(); if(mediaEl) mediaEl.currentTime = parseFloat(endRange.value); });

function syncFromTimers(which){
  const s = parseTs(startTimer.value), e = parseTs(endTimer.value);
  if(which==='start' && s!==null){ startRange.value = Math.min(Math.max(s,0), state.duration); }
  if(which==='end' && e!==null){ endRange.value = Math.min(Math.max(e,0), state.duration); }
  syncFromRanges();
}
startTimer.addEventListener('change', ()=>syncFromTimers('start'));
endTimer.addEventListener('change', ()=>syncFromTimers('end'));

document.getElementById('setStartBtn').addEventListener('click', ()=>{
  if(!mediaEl) return;
  startRange.value = mediaEl.currentTime; syncFromRanges();
});
document.getElementById('setEndBtn').addEventListener('click', ()=>{
  if(!mediaEl) return;
  endRange.value = mediaEl.currentTime; syncFromRanges();
});

document.getElementById('addChunkBtn').addEventListener('click', async ()=>{
  const btn = document.getElementById('addChunkBtn');
  const start = parseFloat(startRange.value), end = parseFloat(endRange.value);
  if(end - start < 0.05){ toast('Select a longer range first.'); return; }
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Cutting…';
  try{
    const data = await api('/api/chunk', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({start, end}),
    });
    state.chunks = data.chunks;
    renderChunkList();
  }catch(e){ toast(e.message); }
  finally{ btn.disabled = false; btn.textContent = 'Add chunk'; }
});

function renderChunkList(){
  const ul = document.getElementById('chunkList');
  ul.innerHTML = '';
  document.getElementById('chunkEmptyNote').classList.toggle('hidden', state.chunks.length>0);

  const downloadAllBtn = document.getElementById('downloadAllChunksBtn');
  if (downloadAllBtn) {
    downloadAllBtn.classList.toggle('hidden', state.chunks.length === 0);
  }

  state.chunks.forEach(c=>{
    const li = document.createElement('li');
    li.innerHTML = `<span><span class="idx">#${c.id}</span> <span class="meta">${fmtTs(c.start)} – ${fmtTs(c.end)} (${c.duration.toFixed(2)}s)</span></span>`;

    const btnContainer = document.createElement('div');
    btnContainer.style.display = 'flex';
    btnContainer.style.gap = '6px';

    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn secondary small';
    downloadBtn.textContent = 'Download';
    downloadBtn.title = 'Download this chunk';
    downloadBtn.addEventListener('click', ()=>{
      window.location.href = `/api/chunk/${c.id}/download`;
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'btn danger small';
    delBtn.textContent = 'Remove';
    delBtn.addEventListener('click', async ()=>{
      try{
        const data = await api(`/api/chunk/${c.id}`, {method:'DELETE'});
        state.chunks = data.chunks;
        renderChunkList();
      }catch(e){ toast(e.message); }
    });

    btnContainer.appendChild(downloadBtn);
    btnContainer.appendChild(delBtn);
    li.appendChild(btnContainer);
    ul.appendChild(li);
  });
}

document.getElementById('downloadAllChunksBtn').addEventListener('click', ()=>{
  window.location.href = '/api/chunks/download';
});

// ---------- tab 3: transcription controls ----------
const engineSel = document.getElementById('engineSel');
const deviceSel = document.getElementById('deviceSel');
const diarizeCk = document.getElementById('diarizeCk');
const diarizeFields = document.getElementById('diarizeFields');
const pyannoteCk = document.getElementById('pyannoteCk');
const aiApiCk = document.getElementById('aiApiCk');
const aiApiSub = document.getElementById('aiApiSub');
const apiKeyCk = document.getElementById('apiKeyCk');
const apiKeySub = document.getElementById('apiKeySub');
const apiKeyInput = document.getElementById('apiKeyInput');

function updateDeviceOptions(){
  const mlx = engineSel.value === 'mlx';
  [...deviceSel.options].forEach(o=>{
    o.disabled = mlx ? o.value !== 'metal' : o.value === 'metal';
  });
  deviceSel.value = mlx ? 'metal' : (deviceSel.value === 'metal' ? 'cpu' : deviceSel.value);
  deviceSel.disabled = mlx;
}
engineSel.addEventListener('change', updateDeviceOptions);
updateDeviceOptions();

diarizeCk.addEventListener('change', ()=>{
  diarizeFields.disabled = !diarizeCk.checked;
  if(!diarizeCk.checked){
    pyannoteCk.checked = false; aiApiCk.checked = false;
  }
  updateDiarizeUI();
});

function setMutex(checkedBox, otherBox){
  checkedBox.disabled = false; otherBox.disabled = checkedBox.checked;
}
pyannoteCk.addEventListener('change', ()=>{ if(pyannoteCk.checked) aiApiCk.checked = false; updateDiarizeUI(); });
aiApiCk.addEventListener('change', ()=>{ if(aiApiCk.checked) pyannoteCk.checked = false; updateDiarizeUI(); });

function updateDiarizeUI(){
  setMutex(pyannoteCk, aiApiCk);
  setMutex(aiApiCk, pyannoteCk);
  aiApiSub.classList.toggle('hidden', !aiApiCk.checked);

  const needsKey = pyannoteCk.checked || aiApiCk.checked;
  apiKeyCk.checked = needsKey;
  apiKeySub.classList.toggle('hidden', !needsKey);
  apiKeyInput.placeholder = pyannoteCk.checked ? 'Hugging Face access token' : 'Provider API key';
}
updateDiarizeUI();

document.getElementById('transcribeBtn').addEventListener('click', async ()=>{
  const btn = document.getElementById('transcribeBtn');
  const diarize = diarizeCk.checked;
  if(diarize && !pyannoteCk.checked && !aiApiCk.checked){
    toast('Pick a diarization method: pyannote or AI API.'); return;
  }
  if(diarize && (pyannoteCk.checked || aiApiCk.checked) && !apiKeyInput.value.trim()){
    toast('An API key / token is required for the selected diarization method.'); return;
  }

  const payload = {
    engine: engineSel.value,
    language: document.getElementById('langSel').value,
    device: deviceSel.value,
    model_size: document.getElementById('modelSel').value,
    diarize,
    diarize_method: pyannoteCk.checked ? 'pyannote' : (aiApiCk.checked ? 'ai_api' : null),
    ai_provider: document.getElementById('aiProviderSel').value,
    api_key: apiKeyInput.value.trim(),
  };

  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Transcribing…';
  try{
    const requestOptions = {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload),
    };
    if (state.multifile) {
      goTab('extract');
      state.transcriptions = [];
      renderMultiTranscriptions(state.transcriptions, {
        state:'running', completed:0, total:state.fileCount,
      });
      let requestSettled = false;
      const transcriptionRequest = api('/api/transcribe', requestOptions)
        .then(data => ({data}), error => ({error}))
        .finally(() => { requestSettled = true; });

      let lastCount = -1;
      while (true) {
        const progress = await api('/api/transcribe/status');
        const items = progress.transcriptions || [];
        if (items.length !== lastCount) {
          state.transcriptions = items;
          renderMultiTranscriptions(items, progress.status);
          lastCount = items.length;
        } else {
          updateMultiTranscriptionStatus(progress.status);
        }
        if (progress.status.state === 'complete' || progress.status.state === 'failed') break;
        if (requestSettled && progress.status.state === 'idle') {
          const outcome = await transcriptionRequest;
          if (outcome.error) throw outcome.error;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      const outcome = await transcriptionRequest;
      if (outcome.error) throw outcome.error;
      state.transcriptions = outcome.data.transcriptions || [];
      renderMultiTranscriptions(state.transcriptions, {
        state:'complete', completed:state.transcriptions.length, total:state.transcriptions.length,
      });
    } else {
      const data = await api('/api/transcribe', requestOptions);
      state.transcriptions = [];
      state.segments = data.segments;
      document.getElementById('transcriptOut').value = data.transcript;
      setupSpeakerRenamePanel(data.segments);
      showSingleResults();
      goTab('extract');
    }
  }catch(e){ toast(e.message); }
  finally{ btn.disabled = false; btn.textContent = 'Transcribe'; }
});

// ---------- tab 4: results ----------
const singleResultWrap = document.getElementById('singleResultWrap');
const multiResultWrap = document.getElementById('multiResultWrap');
const multiResultList = document.getElementById('multiResultList');
const multiResultStatus = document.getElementById('multiResultStatus');

function showSingleResults(){
  singleResultWrap.classList.remove('hidden');
  multiResultWrap.classList.add('hidden');
  multiResultList.innerHTML = '';
}
function showMultiResults(){
  singleResultWrap.classList.add('hidden');
  multiResultWrap.classList.remove('hidden');
}
function clearResultsUI(){
  document.getElementById('transcriptOut').value = '';
  document.getElementById('speakerRenamePanel').classList.add('hidden');
  multiResultList.innerHTML = '';
  multiResultStatus.textContent = '';
  state.transcriptions = [];
  showSingleResults();
}

function updateMultiTranscriptionStatus(status){
  if (!status) {
    multiResultStatus.textContent = '';
  } else if (status.state === 'running') {
    multiResultStatus.textContent = status.completed
      ? `Ready: ${status.completed} of ${status.total} files. Transcribing the next file…`
      : `Waiting for the first transcription (${status.total} files)…`;
  } else if (status.state === 'failed') {
    multiResultStatus.textContent =
      `Transcription stopped after ${status.completed} of ${status.total} files. Completed results remain available.`;
  } else if (status.state === 'complete') {
    multiResultStatus.textContent = `All ${status.completed} files transcribed.`;
  } else {
    multiResultStatus.textContent = '';
  }
}

function renderMultiTranscriptions(items, status){
  multiResultList.innerHTML = '';
  if (!items || items.length === 0){
    updateMultiTranscriptionStatus(status);
    showMultiResults();
    return;
  }

  items.forEach((item, idx)=>{
    const wrap = document.createElement('div');
    wrap.className = 'multi-result-item';

    const header = document.createElement('div');
    header.className = 'mr-header';

    const nameEl = document.createElement('div');
    nameEl.className = 'mr-name';
    nameEl.textContent = item.name || `File ${idx+1}`;

    const dlBtn = document.createElement('button');
    dlBtn.className = 'btn secondary small';
    dlBtn.textContent = 'Download';
    dlBtn.title = 'Download this transcription';
    dlBtn.addEventListener('click', ()=>{
      window.location.href = item.id
        ? `/api/download/transcription/file/${encodeURIComponent(item.id)}`
        : `/api/download/transcription/${idx}`;
    });

    header.appendChild(nameEl);
    header.appendChild(dlBtn);

    const ta = document.createElement('textarea');
    ta.readOnly = true;
    ta.value = item.transcript || '';
    ta.spellcheck = false;

    wrap.appendChild(header);
    wrap.appendChild(ta);
    multiResultList.appendChild(wrap);
  });

  updateMultiTranscriptionStatus(status);
  showMultiResults();
}

function setupSpeakerRenamePanel(segments){
  const panel = document.getElementById('speakerRenamePanel');
  const list = document.getElementById('speakerRenameList');

  list.innerHTML = '';

  const speakers = new Set();
  (segments || []).forEach(seg => {
    if (seg.speaker) {
      speakers.add(seg.speaker);
    }
  });

  if (speakers.size <= 1) {
    panel.classList.add('hidden');
    return;
  }

  panel.classList.remove('hidden');
  speakers.forEach(speaker => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; gap:8px; align-items:center; margin-bottom:8px;';
    row.innerHTML = `
      <span style="font-family:var(--font-mono); color:var(--text-faint); font-size:11.5px; width:100px;">${speaker}</span>
      <input type="text" class="speaker-rename-input" data-original="${speaker}" placeholder="Enter name" style="flex:1;">
    `;
    list.appendChild(row);
  });
}

document.getElementById('applySpeakerNamesBtn').addEventListener('click', async ()=>{
  const inputs = document.querySelectorAll('.speaker-rename-input');
  const speakerMapping = {};
  let hasChanges = false;

  inputs.forEach(input => {
    const original = input.dataset.original;
    const newName = input.value.trim();
    if (newName && newName !== original) {
      speakerMapping[original] = newName;
      hasChanges = true;
    }
  });

  if (!hasChanges) {
    toast('No speaker names to apply.');
    return;
  }

  const btn = document.getElementById('applySpeakerNamesBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Applying…';

  try {
    const data = await api('/api/rename_speakers', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({speaker_mapping: speakerMapping}),
    });

    state.segments = data.segments;
    document.getElementById('transcriptOut').value = data.transcript;
    setupSpeakerRenamePanel(data.segments);
    toast('Speaker names applied successfully.');
  } catch(e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Apply names';
  }
});

document.getElementById('downloadTxtBtn').addEventListener('click', ()=>{
  window.location.href = '/api/download/transcription';
});
document.getElementById('downloadZipBtn').addEventListener('click', ()=>{
  window.location.href = '/api/download/zip';
});
document.getElementById('downloadZipBtnMulti').addEventListener('click', ()=>{
  window.location.href = '/api/download/zip';
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    # threaded=True so status polling works while a transcription request runs.
    app.run(debug=True, host="0.0.0.0", port=5030, threaded=True)
