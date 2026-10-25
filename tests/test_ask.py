import anyio
import pytest

from quackback.ask import NO_ANSWER, Asker, AskFailed, spoken_question
from quackback.config import Settings
from tests.fakes import FakeKeyboard, FakeVoice

pytestmark = pytest.mark.anyio

DATABASES = ["PostgreSQL", "SQLite"]


def _asker(speech: FakeVoice | None, keyboard: FakeKeyboard, **settings: object) -> Asker:
    return Asker(Settings(**settings), voice=speech, keyboard=keyboard)  # type: ignore[arg-type]


async def test_voice_answer() -> None:
    voice = FakeVoice(["Ship it on Friday."])
    result = await _asker(voice, FakeKeyboard()).ask("When should we ship?")
    assert result["answer"] == "Ship it on Friday."
    assert result["source"] == "voice"
    assert isinstance(result["duration_s"], float)
    assert "transcript" not in result
    assert voice.said[0] == "Your coding agent has a question. When should we ship?"
    assert voice.said[-1] == "Got it."
    assert voice.beeps == 1


async def test_options_are_read_out_matched_and_transcribed() -> None:
    voice = FakeVoice(["um, the second one"])
    result = await _asker(voice, FakeKeyboard()).ask("Which database?", "For the cache.", DATABASES)
    assert result["answer"] == "SQLite"
    assert result["transcript"] == "um, the second one"
    assert voice.said[0] == (
        "Your coding agent has a question. For the cache. Which database? "
        "Option 1: PostgreSQL. Option 2: SQLite."
    )
    assert voice.hotwords == [DATABASES]
    assert voice.said[-1] == "Got it: SQLite."


async def test_unmatched_answer_with_options_is_the_transcript() -> None:
    voice = FakeVoice(["neither, use a flat file"])
    result = await _asker(voice, FakeKeyboard()).ask("Which database?", None, DATABASES)
    assert result["answer"] == "neither, use a flat file"
    assert result["transcript"] == "neither, use a flat file"


async def test_blank_options_are_ignored() -> None:
    voice = FakeVoice(["yes"])
    result = await _asker(voice, FakeKeyboard()).ask("Proceed?", None, ["", "  "])
    assert "transcript" not in result
    assert "Option" not in voice.said[0]


async def test_silence_is_asked_again() -> None:
    voice = FakeVoice(["", "Friday"])
    result = await _asker(voice, FakeKeyboard()).ask("When?")
    assert result == {"answer": "Friday", "source": "voice", "duration_s": result["duration_s"]}
    assert voice.said[1] == "Sorry, I didn't catch that. When?"
    assert voice.beeps == 2


async def test_silence_twice_goes_to_the_keyboard() -> None:
    voice, keyboard = FakeVoice(["", ""]), FakeKeyboard("typed")
    result = await _asker(voice, keyboard).ask("When?")
    assert result["source"] == "keyboard"
    assert result["answer"] == "typed"
    assert voice.said[-1] == "I'll show the question on your screen."


async def test_voice_error_goes_to_the_keyboard() -> None:
    keyboard = FakeKeyboard("typed")
    result = await _asker(FakeVoice(fail=True), keyboard).ask("When?", "ctx", ["a"])
    assert result["source"] == "keyboard"
    assert keyboard.asked == [("When?", "ctx", ["a"])]


async def test_voice_not_ready_goes_to_the_keyboard() -> None:
    voice = FakeVoice(["unused"], ready=False)
    result = await _asker(voice, FakeKeyboard()).ask("When?")
    assert result["source"] == "keyboard"
    assert voice.said == []


async def test_no_voice_goes_to_the_keyboard() -> None:
    result = await _asker(None, FakeKeyboard("typed")).ask("When?")
    assert result["source"] == "keyboard"


async def test_voice_disabled_goes_to_the_keyboard() -> None:
    voice = FakeVoice(["unused"])
    result = await _asker(voice, FakeKeyboard(), voice=False).ask("When?")
    assert result["source"] == "keyboard"
    assert voice.said == []


async def test_keyboard_answer_is_matched_too() -> None:
    result = await _asker(None, FakeKeyboard("sqlite")).ask("Which?", None, DATABASES)
    assert result["answer"] == "SQLite"
    assert result["transcript"] == "sqlite"


async def test_dismissed_dialog_is_no_answer() -> None:
    result = await _asker(None, FakeKeyboard(None)).ask("When?")
    assert result["source"] == "timeout"
    assert result["answer"] == NO_ANSWER


async def test_unreachable_human_raises() -> None:
    with pytest.raises(AskFailed, match="no display"):
        await _asker(None, FakeKeyboard(unavailable=True)).ask("When?")


async def test_timeout() -> None:
    asker = _asker(None, FakeKeyboard(delay=30), ask_timeout=0.3)
    with anyio.fail_after(5):
        result = await asker.ask("When?")
    assert result["source"] == "timeout"
    assert result["answer"] == "The human did not answer. Make a reasonable decision and note it."
    assert result["duration_s"] == pytest.approx(0.3, abs=0.2)


async def test_timeout_while_listening() -> None:
    asker = _asker(FakeVoice(["late"], listen_delay=30), FakeKeyboard(), ask_timeout=0.3)
    with anyio.fail_after(5):
        result = await asker.ask("When?")
    assert result["source"] == "timeout"


async def test_progress_while_waiting() -> None:
    reports: list[tuple[float, float, str]] = []
    release = anyio.Event()

    async def progress(elapsed: float, total: float, message: str) -> None:
        reports.append((elapsed, total, message))
        if len(reports) >= 2:
            release.set()  # the human answers once two notifications went out

    asker = _asker(None, FakeKeyboard(release=release), progress_interval=0.05, ask_timeout=60)
    with anyio.fail_after(10):
        result = await asker.ask("When?", progress=progress)
    assert result["source"] == "keyboard"
    assert len(reports) >= 2
    assert all(total == 60 for _, total, _ in reports)
    assert [elapsed for elapsed, _, _ in reports] == sorted(elapsed for elapsed, _, _ in reports)
    assert reports[0][2].startswith("Waiting for the human")


async def test_progress_failure_does_not_break_the_question() -> None:
    async def broken(elapsed: float, total: float, message: str) -> None:
        raise RuntimeError("client gone")

    asker = _asker(None, FakeKeyboard("ok", delay=0.25), progress_interval=0.05)
    result = await asker.ask("When?", progress=broken)
    assert result["answer"] == "ok"


async def test_one_question_at_a_time() -> None:
    keyboard = FakeKeyboard(delay=0.2)
    asker = _asker(None, keyboard)
    results = []

    async def ask(question: str) -> None:
        results.append(await asker.ask(question))

    async with anyio.create_task_group() as tg:
        tg.start_soon(ask, "first")
        tg.start_soon(ask, "second")
    assert keyboard.max_active == 1
    assert len(results) == 2


async def test_timeout_counts_the_wait_for_an_earlier_question() -> None:
    asker = _asker(None, FakeKeyboard(delay=30), ask_timeout=1.0)
    results = []

    async def ask(question: str) -> None:
        results.append(await asker.ask(question))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(ask, "first")
            tg.start_soon(ask, "second")
    assert [r["source"] for r in results] == ["timeout", "timeout"]
    # Both give up after one timeout; the second does not wait for the first and then start over.
    assert all(r["duration_s"] < 1.6 for r in results)


async def test_answer_survives_a_slow_confirmation() -> None:
    voice = FakeVoice(["Friday"], listen_delay=0.1, confirm_delay=1.0)
    asker = _asker(voice, FakeKeyboard(), ask_timeout=0.6)
    with anyio.fail_after(10):
        result = await asker.ask("When?")
    assert result["source"] == "voice"
    assert result["answer"] == "Friday"
    assert result["duration_s"] < 0.6


async def test_silence_without_a_screen_is_no_answer() -> None:
    voice = FakeVoice(["", ""])
    result = await _asker(voice, FakeKeyboard(unavailable=True)).ask("When?")
    assert result["source"] == "timeout"
    assert result["answer"] == NO_ANSWER
    assert "I'll show the question on your screen." not in voice.said


async def test_voice_still_loading_leaves_time_for_the_keyboard() -> None:
    voice = FakeVoice(["unused"], load_time=100)
    keyboard = FakeKeyboard("typed")
    asker = _asker(voice, keyboard, ask_timeout=3)
    with anyio.fail_after(10):
        result = await asker.ask("When?")
    assert result["source"] == "keyboard"
    assert result["duration_s"] < 2.0  # waited a third of the timeout for the voice, not all


async def test_keyboard_is_told_when_to_give_up() -> None:
    keyboard = FakeKeyboard("typed")
    await _asker(None, keyboard, ask_timeout=40).ask("When?")
    assert keyboard.close_after[0] is not None
    assert 39 < keyboard.close_after[0] <= 40


def test_spoken_question_adds_full_stops() -> None:
    assert spoken_question("Ready", "We built it", ["yes", "no!"]) == (
        "Your coding agent has a question. We built it. Ready. Option 1: yes. Option 2: no!"
    )


async def test_real_voice_that_failed_to_load_hands_over_to_the_keyboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quackback.voice import Voice
    from quackback.voice.audio import Audio, AudioUnavailable

    def no_audio(self: Audio) -> None:
        raise AudioUnavailable("no default input device")

    monkeypatch.setattr(Audio, "open", no_audio)
    voice = Voice(Settings())
    voice.load()
    assert voice.error == "AudioUnavailable: no default input device"
    assert not await voice.wait_ready(1)
    result = await Asker(Settings(), voice=voice, keyboard=FakeKeyboard("typed")).ask("Q?")
    assert result["source"] == "keyboard"


async def test_real_voice_still_loading_is_not_ready() -> None:
    from quackback.voice import Voice

    voice = Voice(Settings())  # never loaded: like models still downloading
    with anyio.fail_after(5):
        assert not await voice.wait_ready(0.2)
