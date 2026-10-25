"""Just enough of the sounddevice module for quackback.voice.audio, with no PortAudio."""

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class PortAudioError(Exception):
    pass


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


class WasapiSettings:
    def __init__(self, auto_convert: bool = False) -> None:
        self.auto_convert = auto_convert


class _Default:
    def __init__(self, device: dict[str, int]) -> None:
        self.device = device


class FakeSounddevice:
    """`rates` are the rates a device opens at without help; `host` is the host API name."""

    PortAudioError = PortAudioError
    CallbackStop = CallbackStop
    CallbackAbort = CallbackAbort
    WasapiSettings = WasapiSettings

    def __init__(
        self,
        rates: tuple[int, ...] = (16_000, 22_050, 48_000),
        host: str = "MME",
        default_rate: int = 48_000,
        microphone: Callable[[int, int], np.ndarray] | None = None,
    ) -> None:
        self.rates = rates
        self.host = host
        self.default_rate = default_rate
        # microphone(start_sample, count) -> samples, at the stream's rate
        self.microphone = microphone or (lambda _start, count: np.zeros(count, np.float32))
        self.played: list[tuple[int, np.ndarray]] = []
        self.opened: list[dict[str, Any]] = []
        self.devices: list[dict[str, Any]] = [self._device(0, "fake", hostapi=0)]
        self.hostapis = [host]
        self.default = _Default({"input": 0, "output": 0})

    def _device(self, index: int, name: str, hostapi: int) -> dict[str, Any]:
        return {
            "index": index, "name": name, "hostapi": hostapi, "max_input_channels": 1,
            "max_output_channels": 2, "default_samplerate": float(self.default_rate),
        }  # fmt: skip

    def query_devices(self, device: Any = None, kind: str | None = None) -> Any:
        if device is None and kind is None:
            return self.devices
        if device is None:
            device = self.default.device[kind]
        return self.devices[device]

    def query_hostapis(self, index: int) -> dict[str, Any]:
        return {"name": self.hostapis[index]}

    def _check(self, samplerate: int, extra_settings: Any = None, **_: Any) -> None:
        converts = isinstance(extra_settings, WasapiSettings) and extra_settings.auto_convert
        if samplerate not in self.rates and not converts:
            raise PortAudioError("Invalid sample rate [PaErrorCode -9997]")

    def check_input_settings(self, **kwargs: Any) -> None:
        self._check(**kwargs)

    def check_output_settings(self, **kwargs: Any) -> None:
        self._check(**kwargs)

    def OutputStream(self, **kwargs: Any) -> "_OutputStream":
        self.opened.append(kwargs)
        return _OutputStream(self, **kwargs)

    def InputStream(self, **kwargs: Any) -> "_InputStream":
        self.opened.append(kwargs)
        return _InputStream(self, **kwargs)


class _Stream:
    def __init__(self, samplerate: int, callback: Callable[..., None], **kwargs: Any) -> None:
        self.samplerate = samplerate
        self.callback = callback
        self.finished_callback = kwargs.get("finished_callback")
        self.blocksize = kwargs.get("blocksize") or 512
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.active = False
        self.closed = False

    def start(self) -> None:
        self.active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def abort(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
        self.active = False

    def close(self) -> None:
        self.closed = True

    def _run(self) -> None:
        raise NotImplementedError


class _OutputStream(_Stream):
    def __init__(self, sd: FakeSounddevice, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sd = sd
        self.out: list[np.ndarray] = []

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                buffer = np.full((self.blocksize, 1), np.nan, dtype=np.float32)
                try:
                    self.callback(buffer, self.blocksize, None, None)
                except CallbackStop:
                    self.out.append(buffer[:, 0].copy())
                    break
                except CallbackAbort:
                    break
                self.out.append(buffer[:, 0].copy())
                time.sleep(0.001)
        finally:
            self.sd.played.append((self.samplerate, np.concatenate(self.out or [np.zeros(0)])))
            self.active = False
            if self.finished_callback is not None:
                self.finished_callback()


class _InputStream(_Stream):
    def __init__(self, sd: FakeSounddevice, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sd = sd

    def _run(self) -> None:
        # Like PortAudio, one buffer is reused for every callback: keeping a reference to
        # `indata` instead of copying it would record the last block over and over.
        buffer = np.zeros((self.blocksize, 1), dtype=np.float32)
        position = 0
        while not self._stop.is_set():
            buffer[:, 0] = self.sd.microphone(position, self.blocksize)
            position += self.blocksize
            self.callback(buffer, self.blocksize, None, None)
            time.sleep(0.0005)  # faster than real time; the endpointer counts samples
        self.active = False
