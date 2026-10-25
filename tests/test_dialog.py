import io
import json
import subprocess
import sys

import pytest

from quackback import dialog


def test_alive_and_dead_processes() -> None:
    import os

    assert dialog.process_alive(os.getpid())
    finished = subprocess.run([sys.executable, "-c", "pass"], check=True)
    assert finished.returncode == 0
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not dialog.process_alive(child.pid)


@pytest.mark.parametrize(("answer", "status"), [("Friday", 0), (None, dialog.DISMISSED)])
def test_main_reads_the_request_and_writes_the_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str | None,
    status: int,
) -> None:
    seen: dict[str, object] = {}

    def fake_dialog(question: str, context: str | None, options: list[str], **kw: object):
        seen.update(question=question, context=context, options=options, **kw)
        return answer

    request = {"question": "When?", "context": None, "options": ["a"], "close_after": 3.0,
               "parent_pid": 42}  # fmt: skip
    monkeypatch.setattr(dialog, "run_dialog", fake_dialog)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    assert dialog.main() == status
    assert seen == {"question": "When?", "context": None, "options": ["a"], "close_after": 3.0,
                    "parent_pid": 42}  # fmt: skip
    out = capsys.readouterr().out
    assert (json.loads(out) == {"answer": answer}) if answer else out == ""


def test_dismissed_is_not_a_crash_status() -> None:
    assert dialog.DISMISSED not in (0, 1, 2)
