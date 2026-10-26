"""Text to speech with Piper.

A voice is either a name from rhasspy/piper-voices ("en_US-lessac-medium"), downloaded once into
the models directory, or a path to a local `.onnx` voice with its `.onnx.json` beside it.
"""

import json
import logging
import os
import re
import shutil
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_HF = "https://huggingface.co/"
_TRUE = {"1", "true", "yes", "on"}

_VOICE_NAME = re.compile(r"^[A-Za-z]{2,3}_[A-Za-z]{2}-[A-Za-z0-9_]+-[A-Za-z0-9_]+$")
# Piper reads "[[ ... ]]" as raw phonemes; break up every "[[" so question text never is.
_RAW_PHONEMES = re.compile(r"\[(?=\[)")
# Markdown that agents like to put in questions, and that sounds silly read aloud.
_MARKUP = re.compile(r"[`*#]+")
_GAP_S = 0.15  # pause between sentences
_VOLUME = 0.9
_DOWNLOAD_TIMEOUT_S = 30.0


def is_voice_path(voice: str) -> bool:
    return voice.endswith(".onnx") or "/" in voice or "\\" in voice


# progress(file name, bytes so far, total bytes or None)
Progress = Callable[[str, int, int | None], None]


class LogProgress:
    """Progress as log lines, one per tenth (the server has no terminal to draw a bar on)."""

    def __init__(self) -> None:
        self._last_step: dict[str, int] = {}

    def __call__(self, name: str, done: int, total: int | None) -> None:
        if not total:
            return
        step = done * 10 // total
        if step > self._last_step.get(name, -1):
            self._last_step[name] = step
            log.info("downloading %s: %d%% of %.0f MB", name, step * 10, total / 1e6)


def download(url: str, target: Path, progress: Progress | None = None) -> None:
    """Download to `target` through `target.part`, picking up where an earlier attempt stopped."""
    partial = target.with_name(target.name + ".part")
    have = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=_DOWNLOAD_TIMEOUT_S
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # the partial file is not a prefix of this file: start over
            partial.unlink()
            download(url, target, progress)
            return
        raise
    with response:
        if have and response.status != 206:  # the server ignored the range: start over
            have = 0
        length = response.headers.get("Content-Length")
        total = have + int(length) if length and length.isdigit() else None
        done = have
        with partial.open("ab" if have else "wb") as out:
            while chunk := response.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(target.name, done, total)
    os.replace(partial, target)


def ensure_voice(voice: str, voices_dir: Path, progress: Progress | None = None) -> Path:
    """Return the local model for a voice name, downloading it if needed. Blocking."""
    from piper.download_voices import URL_FORMAT, VOICE_PATTERN

    voice = voice.strip()
    match = VOICE_PATTERN.match(voice)
    if not match or not _VOICE_NAME.match(voice):
        raise ValueError(f"not a Piper voice name: {voice!r}")
    model = voices_dir / f"{voice}.onnx"
    config = voices_dir / f"{voice}.onnx.json"
    if all(path.is_file() and path.stat().st_size > 0 for path in (model, config)):
        for leftover in (model, config):  # a download interrupted after an earlier one finished
            leftover.with_name(leftover.name + ".part").unlink(missing_ok=True)
        return model
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE:
        raise FileNotFoundError(f"Piper voice {voice} is not downloaded and HF_HUB_OFFLINE is set")
    voices_dir.mkdir(parents=True, exist_ok=True)
    family = match.group("lang_family")
    parts = {
        "lang_family": family,
        "lang_code": f"{family}_{match.group('lang_region')}",
        "voice_name": match.group("voice_name"),
        "voice_quality": match.group("voice_quality"),
    }
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    log.info("downloading Piper voice %s", voice)
    # The model goes last: once it is in place, the voice is complete. A download cut short
    # leaves a .part file that the next attempt continues.
    for target, extension in ((config, ".onnx.json"), (model, ".onnx")):
        if target.is_file() and target.stat().st_size > 0:
            continue
        url = URL_FORMAT.format(extension=extension, **parts)
        if endpoint and url.startswith(_HF):
            url = f"{endpoint}/{url.removeprefix(_HF)}"
        download(url, target, progress)
    return model


def espeak_data_dir(
    source: Path | None = None,
    platform: str = sys.platform,
    program_data: Path | None = None,
) -> Path:
    """espeak-ng's data directory, at a path espeak-ng can actually open.

    espeak-ng opens its files through narrow C strings and calls exit() when it cannot, which
    kills the whole server. On Windows that happens for any non-ASCII path, such as a user name
    in Cyrillic inside the install directory. Use the 8.3 short path, or else a copy under
    ProgramData; failing both, raise, so voice is off but the server stays up.
    """
    if source is None:
        from piper.phonemize_espeak import ESPEAK_DATA_DIR

        source = Path(ESPEAK_DATA_DIR)
    if platform != "win32" or str(source).isascii():
        return source
    short = _short_path(source)
    if short is not None and str(short).isascii():
        return short
    base = program_data or Path(os.environ.get("PROGRAMDATA") or r"C:\ProgramData")
    target = base / "quackback" / f"espeak-ng-data-{version('piper-tts')}"
    if not str(target).isascii():
        raise RuntimeError(f"espeak-ng cannot read its data from a non-ASCII path: {source}")
    marker = target / ".complete"
    if not marker.is_file():
        log.info("copying espeak-ng data to %s (the install path is not ASCII)", target)
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        marker.touch()
    return target


def _short_path(path: Path) -> Path | None:
    """The Windows 8.3 short form of a path, if the volume keeps them."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    get_short = ctypes.windll.kernel32.GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD
    size = get_short(str(path), None, 0)
    if size == 0:
        return None
    buffer = ctypes.create_unicode_buffer(size)
    if get_short(str(path), buffer, size) == 0:
        return None
    return Path(buffer.value)


class TextToSpeech:
    def __init__(self, voice: str, voices_dir: Path) -> None:
        self._voice_name = voice
        self._voices_dir = voices_dir
        self._voice: Any = None
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        if self._voice is None:
            raise RuntimeError("the voice is not loaded")
        return int(self._voice.config.sample_rate)

    def model_path(self, progress: Progress | None = None) -> Path:
        """The voice model on disk, downloading a named voice first. Blocking."""
        if is_voice_path(self._voice_name):
            model = Path(self._voice_name).expanduser()
            if not model.is_file():
                raise FileNotFoundError(f"Piper voice not found: {model}")
            return model
        return ensure_voice(self._voice_name, self._voices_dir, progress)

    def load(self) -> None:
        """Load the voice and warm it up (espeak start-up, first inference). Blocking."""
        import onnxruntime
        from piper import PiperConfig, PiperVoice

        model = self.model_path(LogProgress())
        with open(f"{model}.json", encoding="utf-8") as f:
            config = PiperConfig.from_dict(json.load(f))
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3  # errors only
        # enable_fallback=0: on a provider error onnxruntime would otherwise print to stdout.
        session = onnxruntime.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"], enable_fallback=0
        )
        self._voice = PiperVoice(session=session, config=config, espeak_data_dir=espeak_data_dir())
        self.synthesize("Ready.")
        log.info("Piper voice %s loaded (%d Hz)", model.name, self.sample_rate)

    def synthesize(self, text: str) -> np.ndarray:
        """Mono float32 audio at `sample_rate`. Blocking; call from a worker thread."""
        from piper import SynthesisConfig

        if self._voice is None:
            raise RuntimeError("the voice is not loaded")
        text = speakable(text)
        gap = np.zeros(int(self.sample_rate * _GAP_S), dtype=np.float32)
        parts: list[np.ndarray] = []
        with self._lock:
            for chunk in self._voice.synthesize(text, syn_config=SynthesisConfig(volume=_VOLUME)):
                if parts:
                    parts.append(gap)
                parts.append(chunk.audio_float_array)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def speakable(text: str) -> str:
    """Question text as it should be read aloud."""
    return " ".join(_MARKUP.sub("", _RAW_PHONEMES.sub("[ ", text)).split())
