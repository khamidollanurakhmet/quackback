"""Speech on the PC: Piper speaks, faster-whisper listens, sounddevice moves the audio.

`Voice.start_loading()` loads everything on a background thread at server start (Whisper once,
not per question). Until it is ready, or if it failed, questions go to the keyboard instead.
"""

import logging
import threading
from collections.abc import Sequence

import anyio
import anyio.to_thread

from quackback.config import Settings, default_cache_dir
from quackback.voice.audio import Audio
from quackback.voice.stt import SpeechToText
from quackback.voice.tts import TextToSpeech

log = logging.getLogger(__name__)


class Voice:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        voices_dir = (settings.models_dir or default_cache_dir()) / "voices"
        self.tts = TextToSpeech(settings.piper_voice, voices_dir)
        self.stt = SpeechToText(
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
            settings.language,
        )
        self.audio = Audio(settings.input_device, settings.output_device)
        self._loaded = threading.Event()
        self.error: str | None = None

    def start_loading(self) -> None:
        threading.Thread(target=self.load, name="quackback-voice-load", daemon=True).start()

    def load(self) -> None:
        """Open audio, load Piper and Whisper. Blocking; records the failure instead of raising."""
        try:
            self.audio.open()
            self.tts.load()
            self.stt.load()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("voice unavailable, questions will use the keyboard: %s", self.error)
            log.debug("voice load failure", exc_info=True)
        finally:
            self._loaded.set()

    async def wait_ready(self, max_wait: float) -> bool:
        loaded = await anyio.to_thread.run_sync(self._loaded.wait, max_wait, abandon_on_cancel=True)
        return loaded and self.error is None

    async def say(self, text: str) -> None:
        samples = await anyio.to_thread.run_sync(self.tts.synthesize, text, abandon_on_cancel=True)
        await self.audio.play(samples, self.tts.sample_rate)

    async def beep(self) -> None:
        await self.audio.beep()

    async def listen(self, hotwords: Sequence[str] = ()) -> str:
        s = self._settings
        audio = await self.audio.record_answer(s.listen_timeout, s.end_silence, s.max_answer)
        if audio is None:
            return ""
        return await anyio.to_thread.run_sync(
            self.stt.transcribe, audio, hotwords, abandon_on_cancel=True
        )
