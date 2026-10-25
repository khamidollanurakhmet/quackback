"""With the real models: Piper speaks, the endpointer hears it, Whisper writes it down.

Opt-in (downloads ~500 MB the first time): QUACKBACK_TEST_MODELS=1 uv run pytest -m models
"""

import os

import numpy as np
import pytest

from quackback.config import Settings, default_cache_dir
from quackback.options import match_option
from quackback.voice.audio import SPEECH_RATE, resample
from quackback.voice.endpoint import Endpointer, State
from quackback.voice.stt import SpeechToText
from quackback.voice.tts import TextToSpeech

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("QUACKBACK_TEST_MODELS") != "1", reason="set QUACKBACK_TEST_MODELS=1"
    ),
]


@pytest.fixture(scope="module")
def tts() -> TextToSpeech:
    settings = Settings()
    engine = TextToSpeech(settings.piper_voice, default_cache_dir() / "voices")
    engine.load()
    return engine


@pytest.fixture(scope="module")
def stt() -> SpeechToText:
    settings = Settings()
    engine = SpeechToText(settings.whisper_model, "cpu", "int8", "en")
    engine.load()
    return engine


def _spoken(tts: TextToSpeech, text: str) -> np.ndarray:
    return resample(tts.synthesize(text), tts.sample_rate, SPEECH_RATE)


def test_loopback_free_answer(tts: TextToSpeech, stt: SpeechToText) -> None:
    text = stt.transcribe(_spoken(tts, "Ship it on Friday after the tests pass."))
    words = text.lower()
    assert "friday" in words
    assert "test" in words


def test_loopback_option(tts: TextToSpeech, stt: SpeechToText) -> None:
    options = ["PostgreSQL", "SQLite", "Redis"]
    text = stt.transcribe(_spoken(tts, "Let's go with SQLite."), hotwords=options)
    assert match_option(text, options) == "SQLite"


def test_loopback_ordinal(tts: TextToSpeech, stt: SpeechToText) -> None:
    options = ["PostgreSQL", "SQLite", "Redis"]
    text = stt.transcribe(_spoken(tts, "The third one."), hotwords=options)
    assert match_option(text, options) == "Redis"


def test_endpointer_hears_real_speech(tts: TextToSpeech) -> None:
    rng = np.random.default_rng(0)
    hiss = (rng.standard_normal(SPEECH_RATE) * 0.002).astype(np.float32)
    signal = np.concatenate([hiss, _spoken(tts, "Yes, go ahead."), hiss, hiss])
    endpointer = Endpointer(SPEECH_RATE, listen_timeout=3, end_silence=1.2, max_answer=30)
    block = SPEECH_RATE // 20
    for start in range(0, len(signal), block):
        if endpointer.feed(signal[start : start + block]) is State.DONE:
            break
    assert endpointer.state is State.DONE


def test_silence_transcribes_to_nothing(stt: SpeechToText) -> None:
    rng = np.random.default_rng(1)
    hiss = (rng.standard_normal(SPEECH_RATE * 2) * 0.002).astype(np.float32)
    assert stt.transcribe(hiss) == ""
