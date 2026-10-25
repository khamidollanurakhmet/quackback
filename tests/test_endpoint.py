import numpy as np
import pytest

from quackback.voice.endpoint import Endpointer, State

RATE = 16_000
BLOCK = 0.03  # seconds


def _blocks(seconds: float, amplitude: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    count = round(seconds / BLOCK)
    size = round(BLOCK * RATE)
    return [(rng.standard_normal(size) * amplitude).astype(np.float32) for _ in range(count)]


def _run(endpointer: Endpointer, *segments: list[np.ndarray]) -> tuple[State, float]:
    """Feed segments in order; return the final state and when it was reached (seconds)."""
    elapsed = 0.0
    for block in (b for segment in segments for b in segment):
        elapsed += BLOCK
        if endpointer.feed(block) in (State.DONE, State.NO_SPEECH):
            return endpointer.state, elapsed
    return endpointer.state, elapsed


def _endpointer(**overrides: float) -> Endpointer:
    settings = {"listen_timeout": 3.0, "end_silence": 1.0, "max_answer": 10.0} | overrides
    return Endpointer(sample_rate=RATE, **settings)


def test_speech_then_silence_is_done() -> None:
    state, at = _run(_endpointer(), _blocks(0.5, 0.001), _blocks(1.5, 0.1), _blocks(2.0, 0.001))
    assert state is State.DONE
    assert at == pytest.approx(0.5 + 1.5 + 1.0, abs=0.1)


def test_short_pauses_do_not_end_the_answer() -> None:
    quiet, loud = 0.001, 0.1
    state, at = _run(
        _endpointer(),
        _blocks(0.5, quiet),
        _blocks(1.0, loud),
        _blocks(0.6, quiet),  # a breath, shorter than end_silence
        _blocks(1.0, loud),
        _blocks(2.0, quiet),
    )
    assert state is State.DONE
    assert at == pytest.approx(0.5 + 1.0 + 0.6 + 1.0 + 1.0, abs=0.1)


def test_nothing_said_times_out() -> None:
    state, at = _run(_endpointer(listen_timeout=2.0), _blocks(5.0, 0.001))
    assert state is State.NO_SPEECH
    assert at == pytest.approx(2.0, abs=0.1)


def test_a_click_is_not_speech() -> None:
    state, _ = _run(
        _endpointer(listen_timeout=2.0),
        _blocks(0.5, 0.001),
        _blocks(0.09, 0.2),  # shorter than min_speech
        _blocks(3.0, 0.001),
    )
    assert state is State.NO_SPEECH


def test_noise_floor_adapts_to_a_loud_room() -> None:
    # A steady hum at 0.05 RMS is the floor; talking at the same level is not speech.
    state, _ = _run(_endpointer(listen_timeout=2.0), _blocks(3.0, 0.05))
    assert state is State.NO_SPEECH
    # Speech well above the hum is.
    endpointer = _endpointer()
    state, _ = _run(endpointer, _blocks(0.5, 0.05), _blocks(1.0, 0.5), _blocks(2.0, 0.05, seed=1))
    assert state is State.DONE


def test_long_answer_is_capped() -> None:
    state, at = _run(_endpointer(max_answer=2.0), _blocks(0.5, 0.001), _blocks(5.0, 0.1))
    assert state is State.DONE
    assert at == pytest.approx(0.5 + 2.0, abs=0.1)


def test_finished_endpointer_stays_finished() -> None:
    endpointer = _endpointer(listen_timeout=1.0)
    _run(endpointer, _blocks(2.0, 0.001))
    assert endpointer.feed(_blocks(0.1, 0.5)[0]) is State.NO_SPEECH
