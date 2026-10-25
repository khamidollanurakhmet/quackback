import threading

import anyio
import numpy as np
import pytest

from quackback.voice.audio import (
    SPEECH_RATE,
    Audio,
    AudioUnavailable,
    _device_index,
    device_spec,
    resample,
    tone,
)
from tests.fake_sounddevice import FakeSounddevice, WasapiSettings


def _audio(sd: FakeSounddevice) -> Audio:
    audio = Audio()
    audio._sd = sd  # skip importing the real sounddevice
    return audio


def _voice_like(rate: int, quiet: float, loud: float, after: float) -> object:
    """A microphone: `quiet` seconds of hiss, `loud` seconds of "speech", then hiss forever."""
    rng = np.random.default_rng(0)

    def microphone(start: int, count: int) -> np.ndarray:
        t = (start + np.arange(count)) / rate
        level = np.where((t >= quiet) & (t < quiet + loud), 0.2, 0.001)
        return (rng.standard_normal(count) * level).astype(np.float32)

    return microphone


def test_device_spec() -> None:
    assert device_spec(None) is None
    assert device_spec(" 3 ") == 3
    assert device_spec("USB Headset") == "USB Headset"


def test_tone_is_short_quiet_and_fades() -> None:
    beep = tone()
    assert beep.dtype == np.float32
    assert 0.1 < len(beep) / 22_050 < 0.2
    assert np.max(np.abs(beep)) <= 0.31
    assert abs(beep[0]) < 0.01
    assert abs(beep[-1]) < 0.01


def test_resample_lengths_and_identity() -> None:
    signal = np.sin(np.linspace(0, 20, 48_000)).astype(np.float32)
    assert len(resample(signal, 48_000, 16_000)) == 16_000
    assert len(resample(signal, 16_000, 48_000)) == 144_000
    assert np.array_equal(resample(signal, 16_000, 16_000), signal)
    assert resample(np.zeros(0, np.float32), 48_000, 16_000).size == 0


def test_format_uses_wanted_rate_when_supported() -> None:
    audio = _audio(FakeSounddevice(rates=(16_000,)))
    assert audio._format("input", 16_000) == (16_000, None)


def test_format_asks_wasapi_to_convert() -> None:
    audio = _audio(FakeSounddevice(rates=(48_000,), host="Windows WASAPI"))
    rate, extra = audio._format("input", 16_000)
    assert rate == 16_000
    assert isinstance(extra, WasapiSettings)
    assert extra.auto_convert


def test_format_falls_back_to_the_device_rate() -> None:
    audio = _audio(FakeSounddevice(rates=(48_000,), host="Windows WDM-KS"))
    assert audio._format("output", 22_050) == (48_000, None)


@pytest.mark.anyio
async def test_play_delivers_every_sample() -> None:
    sd = FakeSounddevice()
    clip = np.linspace(-0.5, 0.5, 5_000, dtype=np.float32)
    await _audio(sd).play(clip, 22_050)
    rate, played = sd.played[-1]
    assert rate == 22_050
    assert np.allclose(played[: len(clip)], clip)
    assert np.all(played[len(clip) :] == 0)


@pytest.mark.anyio
async def test_play_resamples_for_a_fussy_device() -> None:
    sd = FakeSounddevice(rates=(48_000,), host="Windows WDM-KS")
    await _audio(sd).play(np.ones(22_050, np.float32) * 0.1, 22_050)
    rate, played = sd.played[-1]
    assert rate == 48_000
    assert np.count_nonzero(played) == pytest.approx(48_000, rel=0.01)


@pytest.mark.anyio
async def test_cancelled_play_stops() -> None:
    sd = FakeSounddevice()
    long_clip = np.ones(22_050 * 600, np.float32) * 0.1  # ten minutes
    with anyio.fail_after(5), anyio.move_on_after(0.2):
        await _audio(sd).play(long_clip, 22_050)
    for _ in range(50):  # the stream thread finishes right after the abort
        if sd.played:
            break
        await anyio.sleep(0.05)
    assert sd.played
    assert len(sd.played[-1][1]) < len(long_clip)


@pytest.mark.anyio
@pytest.mark.parametrize("rate", [16_000, 48_000])
async def test_record_answer(rate: int) -> None:
    sd = FakeSounddevice(rates=(rate,), default_rate=rate)
    sd.microphone = _voice_like(rate, quiet=0.5, loud=1.0, after=0)  # type: ignore[assignment]
    audio = _audio(sd)
    with anyio.fail_after(10):
        recorded = await audio.record_answer(listen_timeout=3, end_silence=0.8, max_answer=10)
    assert recorded is not None
    seconds = len(recorded) / SPEECH_RATE
    assert seconds == pytest.approx(0.5 + 1.0 + 0.8, abs=0.15)
    assert sd.opened[-1]["channels"] == 1

    def rms(start_s: float, end_s: float) -> float:
        window = recorded[int(start_s * SPEECH_RATE) : int(end_s * SPEECH_RATE)]
        return float(np.sqrt(np.mean(np.square(window))))

    assert rms(0.0, 0.3) < 0.01  # the hiss before the answer
    assert rms(0.8, 1.2) > 0.1  # the answer itself
    assert rms(1.7, 2.1) < 0.01  # the silence after it


@pytest.mark.anyio
async def test_record_silence_is_none() -> None:
    audio = _audio(FakeSounddevice())
    with anyio.fail_after(10):
        assert await audio.record_answer(listen_timeout=1, end_silence=0.8, max_answer=10) is None


@pytest.mark.anyio
async def test_dead_microphone_raises() -> None:
    sd = FakeSounddevice()
    audio = _audio(sd)
    original = sd.InputStream

    def dying_stream(**kwargs: object):
        stream = original(**kwargs)
        stream._run = lambda: setattr(stream, "active", False)  # type: ignore[method-assign]
        return stream

    sd.InputStream = dying_stream  # type: ignore[method-assign]
    with anyio.fail_after(10), pytest.raises(AudioUnavailable, match="stopped"):
        await audio.record_answer(listen_timeout=1, end_silence=0.8, max_answer=10)


def test_audio_runs_on_one_thread() -> None:
    sd = FakeSounddevice()
    audio = _audio(sd)
    threads: set[int] = set()
    original = sd.OutputStream

    def tracking(**kwargs: object):
        threads.add(threading.get_ident())
        return original(**kwargs)

    sd.OutputStream = tracking  # type: ignore[method-assign]

    async def play_twice() -> None:
        await audio.play(np.zeros(100, np.float32), 22_050)
        await audio.play(np.zeros(100, np.float32), 22_050)

    anyio.run(play_twice)
    assert len(threads) == 1
    assert threading.get_ident() not in threads


def test_device_name_prefers_the_default_host_api() -> None:
    sd = FakeSounddevice()
    sd.hostapis = ["MME", "Windows DirectSound", "Windows WASAPI"]
    sd.devices = [
        sd._device(0, "Microsoft Sound Mapper", hostapi=0),
        sd._device(1, "Headphone (Realtek(R) Audio)", hostapi=0),
        sd._device(2, "Headphone (Realtek(R) Audio)", hostapi=1),
        sd._device(3, "Headphone (Realtek(R) Audio)", hostapi=2),
    ]
    sd.default.device = {"input": 0, "output": 0}
    assert _device_index(sd, "output", "headphone") == 1
    sd.default.device = {"input": 3, "output": 3}
    assert _device_index(sd, "output", "Headphone (Realtek(R) Audio)") == 3


def test_unknown_device_name() -> None:
    with pytest.raises(AudioUnavailable, match="no input device named"):
        _device_index(FakeSounddevice(), "input", "Studio Mic 3000")


def test_check_device_resolves_names_and_defaults() -> None:
    sd = FakeSounddevice()
    audio = Audio()
    assert audio._check_device(sd, "input", "fake") == 0
    assert audio._check_device(sd, "input", None) is None
    sd.default.device = {"input": -1, "output": 0}
    with pytest.raises(AudioUnavailable, match="no default input device"):
        audio._check_device(sd, "input", None)
