import logging
from pathlib import Path

import pytest

from quackback import cli
from quackback.config import Settings


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("quackback ")


@pytest.mark.parametrize(
    ("client_ms", "warns"),
    [("600000", False), ("305000", False), ("300000", True), ("60000", True), ("", False)],
)
def test_client_timeout_warning(client_ms: str, warns: bool) -> None:
    warning = cli.check_client_timeout(Settings(ask_timeout=300), {"MCP_TOOL_TIMEOUT": client_ms})
    assert (warning is not None) == warns
    if warning:
        assert "at least 305000" in warning


def test_log_file_directory_is_created(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "deep" / "quackback.log"
    try:
        cli.setup_logging(Settings(log_file=log_file))
        logging.getLogger("quackback.test").warning("hello file")
        assert "hello file" in log_file.read_text(encoding="utf-8")
    finally:
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)


def test_unusable_log_file_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(SystemExit) as exit_info:
        cli.setup_logging(Settings(log_file=blocker / "quackback.log"))
    assert exit_info.value.code == 2
    assert "cannot open the log file" in capsys.readouterr().err


def test_download_failure_is_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from quackback.voice.tts import TextToSpeech

    def offline(self: TextToSpeech) -> Path:
        raise OSError("network is unreachable")

    monkeypatch.setattr(TextToSpeech, "model_path", offline)
    with pytest.raises(SystemExit) as exit_info:
        cli.download(Settings(models_dir=tmp_path))
    assert exit_info.value.code == 1
    assert "download failed: OSError: network is unreachable" in capsys.readouterr().err


def test_download_skips_a_local_whisper_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from quackback.voice.tts import TextToSpeech

    monkeypatch.setattr(TextToSpeech, "model_path", lambda self: tmp_path / "voice.onnx")
    cli.download(Settings(models_dir=tmp_path, whisper_model=str(tmp_path)))
    assert capsys.readouterr().out.rstrip().endswith("done")


def test_setup_sandbox_runs_playwright_through_this_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    import sys
    import types

    ran: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[bytes]:
        ran.append(command)
        return subprocess.CompletedProcess(command, 0)

    # The "sandbox" extra may not be installed here; only its presence is checked.
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["setup-sandbox"])
    assert exit_info.value.code == 0
    assert ran == [[sys.executable, "-m", "playwright", "install", "chromium", "ffmpeg"]]


def test_setup_sandbox_without_the_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "playwright", None)  # makes the import fail
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["setup-sandbox"])
    assert exit_info.value.code == 1
    assert "quackback[sandbox]" in capsys.readouterr().err
