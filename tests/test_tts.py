from pathlib import Path

import pytest

from quackback.voice.tts import TextToSpeech, ensure_voice, is_voice_path, speakable


def test_speakable_strips_markdown_and_raw_phonemes() -> None:
    assert speakable("Should I edit `config.py`?") == "Should I edit config.py?"
    assert speakable("**Really** delete #3?") == "Really delete 3?"
    # "[[" would make Piper read raw phonemes; no "[[" may survive.
    assert "[[" not in speakable("Use [[[x]]] here")
    assert speakable("  two \n lines ") == "two lines"


def test_voice_names_and_paths() -> None:
    assert not is_voice_path("en_US-lessac-medium")
    assert is_voice_path("voices/en_US-lessac-medium.onnx")
    assert is_voice_path(r"C:\voices\custom")


@pytest.mark.parametrize("name", ["lessac", "en_US-../../etc-medium", "en_US-a/b-medium", ""])
def test_bad_voice_names_are_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="Piper voice"):
        ensure_voice(name, tmp_path)


def test_downloaded_voice_is_reused(tmp_path: Path) -> None:
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"model")
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text("{}")
    assert ensure_voice("en_US-lessac-medium", tmp_path) == tmp_path / "en_US-lessac-medium.onnx"


def test_missing_local_voice(tmp_path: Path) -> None:
    tts = TextToSpeech(str(tmp_path / "nope.onnx"), tmp_path)
    with pytest.raises(FileNotFoundError):
        tts.model_path()


def test_synthesize_before_load_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        TextToSpeech("en_US-lessac-medium", tmp_path).synthesize("hi")


def test_espeak_dir_is_used_as_is_when_ascii(tmp_path: Path) -> None:
    from quackback.voice.tts import espeak_data_dir

    assert espeak_data_dir(tmp_path, platform="win32") == tmp_path
    assert espeak_data_dir(tmp_path, platform="linux") == tmp_path


def test_espeak_dir_with_non_ascii_path_is_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quackback.voice import tts

    source = tmp_path / "Нурахмет" / "espeak-ng-data"
    (source / "lang").mkdir(parents=True)
    (source / "phontab").write_bytes(b"data")
    monkeypatch.setattr(tts, "_short_path", lambda path: None)  # 8.3 names turned off
    copied = tts.espeak_data_dir(source, platform="win32", program_data=tmp_path / "pd")
    assert str(copied).isascii()
    assert (copied / "phontab").read_bytes() == b"data"
    assert (copied / "lang").is_dir()
    # Reused the second time, not copied again.
    (copied / "phontab").write_bytes(b"kept")
    assert tts.espeak_data_dir(source, platform="win32", program_data=tmp_path / "pd") == copied
    assert (copied / "phontab").read_bytes() == b"kept"


def test_espeak_dir_short_path_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quackback.voice import tts

    source = tmp_path / "été" / "espeak-ng-data"
    monkeypatch.setattr(tts, "_short_path", lambda path: tmp_path / "ETE~1" / "espeak-ng-data")
    assert tts.espeak_data_dir(source, platform="win32") == tmp_path / "ETE~1" / "espeak-ng-data"


def test_espeak_dir_without_any_ascii_place_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quackback.voice import tts

    monkeypatch.setattr(tts, "_short_path", lambda path: None)
    with pytest.raises(RuntimeError, match="non-ASCII"):
        tts.espeak_data_dir(tmp_path / "été", platform="win32", program_data=tmp_path / "é")


def test_offline_without_a_downloaded_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="HF_HUB_OFFLINE"):
        ensure_voice("en_US-lessac-medium", tmp_path)
