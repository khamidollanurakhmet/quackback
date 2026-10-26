"""Speech to text with faster-whisper. The model is loaded once, in the background, at startup."""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # what Whisper expects (faster_whisper exports no constant)


class SpeechToText:
    def __init__(self, model: str, device: str, compute_type: str, language: str) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language or None
        self._model: Any = None

    def download(self) -> None:
        """Fetch the model into the Hugging Face cache without loading it. Blocking."""
        if os.path.isdir(self._model_name):
            return  # a local model directory: nothing to fetch
        from faster_whisper.utils import download_model

        download_model(self._model_name)

    def is_downloaded(self) -> bool:
        if os.path.isdir(self._model_name):
            return True
        from faster_whisper.utils import download_model

        try:
            download_model(self._model_name, local_files_only=True)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def cache_dir(self) -> Path | None:
        """The Hugging Face cache folder this model downloads into, if it can be told.

        faster-whisper silences the hub's progress bars; watching this folder grow is how a
        terminal can still show that the download is moving.
        """
        try:
            from faster_whisper.utils import _MODELS
            from huggingface_hub import constants
        except ImportError:
            return None
        repo = self._model_name if "/" in self._model_name else _MODELS.get(self._model_name)
        if not repo:
            return None
        return Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))

    def load(self) -> None:
        """Load and warm up the model; fall back to CPU if a GPU setup fails. Blocking."""
        import ctranslate2

        ctranslate2.set_log_level(logging.ERROR)
        try:
            self._model = self._build(self._device, self._compute_type)
        except (RuntimeError, ValueError):
            if self._device == "cpu":
                raise
            # device="cuda"/"auto" without the CUDA libraries fails only once decoding starts
            # (RuntimeError); a build without CUDA at all refuses the device (ValueError).
            log.warning("Whisper on %s failed, using the CPU instead", self._device, exc_info=True)
            self._model = self._build("cpu", "int8")
        from faster_whisper.vad import get_vad_model

        get_vad_model()  # loaded lazily otherwise, on the first answer
        log.info("Whisper model %s loaded", self._model_name)

    def _build(self, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel

        try:  # no network round trip when the model is cached
            model = WhisperModel(
                self._model_name, device=device, compute_type=compute_type, local_files_only=True
            )
        except FileNotFoundError:
            log.info("downloading Whisper model %s", self._model_name)
            model = WhisperModel(self._model_name, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32), language=self._language, beam_size=1,
            temperature=0.0, without_timestamps=True,
        )  # fmt: skip
        list(segments)  # GPU library errors only surface while decoding
        return model

    def transcribe(self, audio: np.ndarray, hotwords: Sequence[str] = ()) -> str:
        """Text of 16 kHz mono float32 speech; "" if there was none. Blocking."""
        from faster_whisper.vad import VadOptions

        if self._model is None:
            raise RuntimeError("the Whisper model is not loaded")
        segments, _ = self._model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=self._language,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=True,
            vad_parameters=VadOptions(min_silence_duration_ms=500),
            without_timestamps=True,
            condition_on_previous_text=False,
            hotwords=", ".join(hotwords) or None,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
