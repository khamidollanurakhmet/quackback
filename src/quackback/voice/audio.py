"""PC audio through sounddevice (PortAudio): play a clip, beep, record a spoken answer.

All PortAudio work runs on one dedicated thread. That serializes device access (sounddevice's
play/wait/stop share global state) and, on Windows, keeps PortAudio's COM setup off the event
loop thread. Streams are stopped from their callbacks, so cancelling an `await` stops the sound
within one block. sounddevice is imported lazily: without PortAudio quackback still runs, keyboard
only.
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import numpy as np

from quackback.voice.endpoint import Endpointer, State

log = logging.getLogger(__name__)

T = TypeVar("T")

# Whisper wants 16 kHz mono float32.
SPEECH_RATE = 16_000
BEEP_RATE = 22_050

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quackback-audio")


class AudioUnavailable(RuntimeError):
    """No usable PortAudio, microphone or speaker."""


def device_spec(value: str | None) -> int | str | None:
    """A sounddevice device: an index if the setting is a number, else a name (substring)."""
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.isdigit() else value


def tone(rate: int = BEEP_RATE, freq: float = 880.0, seconds: float = 0.15) -> np.ndarray:
    """A short sine beep with soft edges, the "speak now" cue."""
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    ramp = min(len(t) // 4, int(rate * 0.01))
    envelope = np.ones_like(wave)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
    envelope[len(t) - ramp :] = np.linspace(1.0, 0.0, ramp)
    return (wave * envelope).astype(np.float32)


def resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resampling, averaging first when downsampling. Good enough for speech."""
    if src == dst or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    if src > dst:
        width = round(src / dst)
        if width > 1:
            samples = np.convolve(samples, np.ones(width) / width, mode="same")
    count = round(samples.size * dst / src)
    positions = np.arange(count) * (src / dst)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def _device_index(sd: Any, kind: str, name: str) -> int:
    """The device whose name contains `name`.

    Windows lists every device once per host API (MME, DirectSound, WASAPI, ...) under the same
    name; prefer the one on the host API of the default device, as sounddevice itself would not.
    """
    wanted = name.lower()
    matches = [
        device for device in sd.query_devices()
        if wanted in device["name"].lower() and device[f"max_{kind}_channels"] > 0
    ]  # fmt: skip
    if not matches:
        raise AudioUnavailable(f"no {kind} device named {name!r}")
    default_host = None
    if sd.default.device[kind] >= 0:
        default_host = sd.query_devices(kind=kind)["hostapi"]
    preferred = [device for device in matches if device["hostapi"] == default_host] or matches
    return int(preferred[0]["index"])


async def _on_audio_thread(fn: Callable[..., T], *args: Any, cancel: threading.Event) -> T:
    """Await `fn(*args)` on the audio thread; on cancellation tell it to stop via `cancel`."""
    try:
        return await asyncio.wrap_future(_EXECUTOR.submit(fn, *args))
    finally:
        cancel.set()


class Audio:
    def __init__(self, input_device: str | None = None, output_device: str | None = None) -> None:
        # A name until _open resolves it to an index.
        self._input: int | str | None = device_spec(input_device)
        self._output: int | str | None = device_spec(output_device)
        self._sd: Any = None
        self._formats: dict[tuple[str, int], tuple[int, Any]] = {}

    def open(self) -> None:
        """Load PortAudio and check the devices. Blocking; raises AudioUnavailable."""
        _EXECUTOR.submit(self._open).result()

    async def play(self, samples: np.ndarray, rate: int) -> None:
        cancel = threading.Event()
        await _on_audio_thread(self._play, samples, rate, cancel, cancel=cancel)

    async def beep(self) -> None:
        await self.play(tone(), BEEP_RATE)

    async def record_answer(
        self, listen_timeout: float, end_silence: float, max_answer: float
    ) -> np.ndarray | None:
        """Record until the answer ends. Returns 16 kHz mono audio, or None if nothing was said."""
        cancel = threading.Event()
        return await _on_audio_thread(
            self._record, listen_timeout, end_silence, max_answer, cancel, cancel=cancel
        )

    # Everything below runs on the audio thread.

    def _open(self) -> None:
        if self._sd is not None:
            return
        try:
            import sounddevice as sd
        except Exception as exc:  # OSError (no PortAudio), PortAudioError, KeyError('PATH'), ...
            raise AudioUnavailable(f"PortAudio is not available: {exc!r}") from exc
        self._input = self._check_device(sd, "input", self._input)
        self._output = self._check_device(sd, "output", self._output)
        self._sd = sd

    def _check_device(self, sd: Any, kind: str, device: int | str | None) -> int | None:
        """The device as an index (None: the default), or AudioUnavailable."""
        try:
            if isinstance(device, str):
                device = _device_index(sd, kind, device)
            elif device is None and sd.default.device[kind] < 0:
                raise AudioUnavailable(f"no default {kind} device")
            info = sd.query_devices(device, kind=kind)
        except AudioUnavailable:
            raise
        except Exception as exc:  # PortAudioError, ValueError (not that kind of device)
            raise AudioUnavailable(f"{kind} device {device!r}: {exc}") from exc
        log.info("audio %s: %s", kind, info["name"])
        return device

    def _format(self, kind: str, wanted: int) -> tuple[int, Any]:
        """(sample rate, extra settings) to open the device with; the caller resamples."""
        if (kind, wanted) in self._formats:
            return self._formats[kind, wanted]
        sd = self._sd
        device = self._input if kind == "input" else self._output
        check = sd.check_input_settings if kind == "input" else sd.check_output_settings
        result: tuple[int, Any] | None = None
        try:
            check(device=device, channels=1, dtype="float32", samplerate=wanted)
            result = (wanted, None)
        except sd.PortAudioError:
            info = sd.query_devices(device, kind=kind)
            # WASAPI refuses rates other than the mixer's unless asked to convert.
            if "WASAPI" in sd.query_hostapis(info["hostapi"])["name"]:
                extra = sd.WasapiSettings(auto_convert=True)
                try:
                    check(device=device, channels=1, dtype="float32", samplerate=wanted,
                          extra_settings=extra)  # fmt: skip
                    result = (wanted, extra)
                except sd.PortAudioError:
                    pass
            if result is None:
                result = (int(info["default_samplerate"]), None)
        self._formats[kind, wanted] = result
        return result

    def _play(self, samples: np.ndarray, rate: int, cancel: threading.Event) -> None:
        sd = self._sd
        if sd is None:
            raise AudioUnavailable("audio is not open")
        device_rate, extra = self._format("output", rate)
        data = resample(np.asarray(samples, dtype=np.float32), rate, device_rate).reshape(-1, 1)
        position = 0
        done = threading.Event()

        def callback(outdata: np.ndarray, frames: int, _time: Any, _status: Any) -> None:
            nonlocal position
            if cancel.is_set():
                outdata.fill(0)
                raise sd.CallbackAbort
            chunk = data[position : position + frames]
            outdata[: len(chunk)] = chunk
            outdata[len(chunk) :] = 0
            position += len(chunk)
            if len(chunk) < frames:
                raise sd.CallbackStop  # play what is queued, then finished_callback

        stream = sd.OutputStream(
            samplerate=device_rate, device=self._output, channels=1, dtype="float32",
            callback=callback, finished_callback=done.set, extra_settings=extra,
        )  # fmt: skip
        try:
            stream.start()
            done.wait(len(data) / device_rate + 2.0)
        finally:
            stream.abort()
            stream.close()

    def _record(
        self, listen_timeout: float, end_silence: float, max_answer: float, cancel: threading.Event
    ) -> np.ndarray | None:
        sd = self._sd
        if sd is None:
            raise AudioUnavailable("audio is not open")
        device_rate, extra = self._format("input", SPEECH_RATE)
        endpointer = Endpointer(device_rate, listen_timeout, end_silence, max_answer)
        blocks: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()

        def callback(indata: np.ndarray, _frames: int, _time: Any, _status: Any) -> None:
            blocks.put(indata[:, 0].copy())  # indata is a view into PortAudio's buffer

        chunks: list[np.ndarray] = []
        deadline = time.monotonic() + listen_timeout + max_answer + 2.0
        stream = sd.InputStream(
            samplerate=device_rate, device=self._input, channels=1, dtype="float32",
            blocksize=max(1, device_rate // 20), callback=callback, extra_settings=extra,
        )  # fmt: skip
        try:
            stream.start()
            while not cancel.is_set() and time.monotonic() < deadline:
                try:
                    block = blocks.get(timeout=0.1)
                except queue.Empty:
                    if not stream.active:  # the device went away
                        raise AudioUnavailable("the microphone stopped") from None
                    continue
                chunks.append(block)
                if endpointer.feed(block) in (State.DONE, State.NO_SPEECH):
                    break
        finally:
            stream.abort()
            stream.close()
        if endpointer.state is not State.DONE:
            return None
        return resample(np.concatenate(chunks), device_rate, SPEECH_RATE)
