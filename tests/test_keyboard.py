import os
import sys
from pathlib import Path

import anyio
import pytest

from quackback.keyboard import KeyboardPrompt, KeyboardUnavailable

pytestmark = pytest.mark.anyio


def _fake_dialog(script: str) -> list[str]:
    """A stand-in for `python -m quackback.dialog` that runs `script` instead of opening Tk."""
    prelude = "import json, sys, time; request = json.load(sys.stdin); "
    return [sys.executable, "-c", prelude + script]


async def test_returns_typed_answer() -> None:
    prompt = KeyboardPrompt(
        _fake_dialog("sys.stdout.write(json.dumps({'answer': request['question'] + '!'}))"),
        notifications=False,
        check_display=False,
    )
    assert await prompt.ask("Ship it", None, []) == "Ship it!"


async def test_passes_context_and_options() -> None:
    prompt = KeyboardPrompt(
        _fake_dialog(
            "sys.stdout.write(json.dumps({'answer': request['context'] + ':' "
            "+ ','.join(request['options'])}))"
        ),
        notifications=False,
        check_display=False,
    )
    assert await prompt.ask("Which?", "We need a DB", ["a", "b"]) == "We need a DB:a,b"


async def test_unicode_survives_the_pipe() -> None:
    prompt = KeyboardPrompt(
        _fake_dialog("sys.stdout.write(json.dumps({'answer': request['question']}))"),
        notifications=False,
        check_display=False,
    )
    assert await prompt.ask("Café — naïve? 🦆", None, []) == "Café — naïve? 🦆"


async def test_closed_window_is_none() -> None:
    prompt = KeyboardPrompt(_fake_dialog("sys.exit(10)"), notifications=False, check_display=False)
    assert await prompt.ask("Q", None, []) is None


async def test_crash_is_unavailable() -> None:
    prompt = KeyboardPrompt(
        _fake_dialog("raise SystemExit('no display')"), notifications=False, check_display=False
    )
    with pytest.raises(KeyboardUnavailable, match="no display"):
        await prompt.ask("Q", None, [])


async def test_garbage_output_is_unavailable() -> None:
    prompt = KeyboardPrompt(
        _fake_dialog("sys.stdout.write('not json')"), notifications=False, check_display=False
    )
    with pytest.raises(KeyboardUnavailable):
        await prompt.ask("Q", None, [])


async def test_missing_executable_is_unavailable() -> None:
    prompt = KeyboardPrompt(
        ["definitely-not-a-real-program-quackback"], notifications=False, check_display=False
    )
    with pytest.raises(KeyboardUnavailable):
        await prompt.ask("Q", None, [])


async def test_cancel_kills_the_dialog() -> None:
    prompt = KeyboardPrompt(
        _fake_dialog("time.sleep(60)"), notifications=False, check_display=False
    )
    with anyio.fail_after(10):  # a leaked 60 s child would blow through this
        with anyio.move_on_after(0.5):
            await prompt.ask("Q", None, [])


def test_dialog_never_imports_from_the_working_directory() -> None:
    from quackback.keyboard import DIALOG_COMMAND

    assert DIALOG_COMMAND[1:] == ("-P", "-m", "quackback.dialog")


async def test_project_files_do_not_shadow_the_dialog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The server runs in the agent's project; a json.py there must not be imported.
    (tmp_path / "json.py").write_text("raise SystemExit('shadowed')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    script = (
        "import json, sys; json.load(sys.stdin); sys.stdout.write(json.dumps({'answer': 'ok'}))"
    )
    prompt = KeyboardPrompt(
        [sys.executable, "-P", "-c", script], notifications=False, check_display=False
    )
    assert await prompt.ask("Q", None, []) == "ok"


async def test_request_carries_a_deadline_and_the_server_pid() -> None:
    prompt = KeyboardPrompt(
        _fake_dialog(
            "sys.stdout.write(json.dumps({'answer': "
            "str(request['close_after']) + ' ' + str(request['parent_pid'])}))"
        ),
        notifications=False,
        check_display=False,
    )
    close_after, pid = (await prompt.ask("Q", None, [], close_after=30.0)).split()
    assert float(close_after) == pytest.approx(32.0)
    assert int(pid) == os.getpid()


async def test_a_slow_notification_does_not_hold_back_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quackback import notify

    async def stuck(title: str, message: str) -> bool:
        await anyio.sleep(30)
        return True

    monkeypatch.setattr(notify, "send", stuck)
    answering = KeyboardPrompt(
        _fake_dialog("sys.stdout.write(json.dumps({'answer': 'fast'}))"), check_display=False
    )
    missing = KeyboardPrompt(["definitely-not-a-real-program-quackback"], check_display=False)
    with anyio.fail_after(10):
        assert await answering.ask("Q", None, []) == "fast"
        with pytest.raises(KeyboardUnavailable):
            await missing.ask("Q", None, [])


def test_available_needs_a_display_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert not KeyboardPrompt().available()
    assert KeyboardPrompt(check_display=False).available()
    monkeypatch.setenv("DISPLAY", ":0")
    assert KeyboardPrompt().available()
