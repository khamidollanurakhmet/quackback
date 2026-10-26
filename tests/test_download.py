"""Voice downloads against a real local HTTP server: progress, resume, and odd servers."""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from quackback.voice.tts import LogProgress, download, ensure_voice

BODY = bytes(range(256)) * 4000  # ~1 MB


class Handler(BaseHTTPRequestHandler):
    honour_ranges = True
    requests: ClassVar[list[str | None]] = []

    def do_GET(self) -> None:
        Handler.requests.append(self.headers.get("Range"))
        wanted = self.headers.get("Range")
        if wanted and self.honour_ranges:
            start = int(wanted.removeprefix("bytes=").rstrip("-"))
            if start >= len(BODY):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            body = BODY[start:]
        else:
            self.send_response(200)
            body = BODY
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    Handler.honour_ranges = True
    Handler.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_downloads_with_progress(server: str, tmp_path: Path) -> None:
    seen: list[tuple[int, int | None]] = []
    target = tmp_path / "voice.onnx"
    download(f"{server}/voice.onnx", target, lambda name, done, total: seen.append((done, total)))
    assert target.read_bytes() == BODY
    assert not (tmp_path / "voice.onnx.part").exists()
    assert seen[-1] == (len(BODY), len(BODY))
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_resumes_an_interrupted_download(server: str, tmp_path: Path) -> None:
    target = tmp_path / "voice.onnx"
    (tmp_path / "voice.onnx.part").write_bytes(BODY[:300_000])
    seen: list[tuple[int, int | None]] = []
    download(f"{server}/voice.onnx", target, lambda name, done, total: seen.append((done, total)))
    assert target.read_bytes() == BODY
    assert Handler.requests == ["bytes=300000-"]
    assert seen[0][0] > 300_000  # counted from where it left off
    assert seen[-1] == (len(BODY), len(BODY))


def test_starts_over_when_the_server_ignores_the_range(server: str, tmp_path: Path) -> None:
    Handler.honour_ranges = False
    target = tmp_path / "voice.onnx"
    (tmp_path / "voice.onnx.part").write_bytes(b"stale bytes")
    download(f"{server}/voice.onnx", target)
    assert target.read_bytes() == BODY


def test_starts_over_when_the_part_is_too_long(server: str, tmp_path: Path) -> None:
    target = tmp_path / "voice.onnx"
    (tmp_path / "voice.onnx.part").write_bytes(BODY + b"extra")  # server answers 416
    download(f"{server}/voice.onnx", target)
    assert target.read_bytes() == BODY
    assert Handler.requests == [f"bytes={len(BODY) + 5}-", None]


def test_ensure_voice_downloads_what_is_missing(
    server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", server)
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text("{}")  # already here
    model = ensure_voice("en_US-lessac-medium", tmp_path)
    assert model.read_bytes() == BODY
    assert (tmp_path / "en_US-lessac-medium.onnx.json").read_text() == "{}"  # left alone
    assert len(Handler.requests) == 1  # only the model was fetched


def test_log_progress_logs_each_tenth_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="quackback.voice.tts")
    progress = LogProgress()
    for done in range(0, 1001, 50):
        progress("voice.onnx", done, 1000)
    lines = [r.message for r in caplog.records]
    assert len(lines) == 11  # 0%, 10%, ..., 100%
    assert lines[-1] == "downloading voice.onnx: 100% of 0 MB"


def test_a_complete_voice_clears_leftover_parts(tmp_path: Path) -> None:
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"model")
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text("{}")
    (tmp_path / "en_US-lessac-medium.onnx.part").write_bytes(b"half a model")
    ensure_voice("en_US-lessac-medium", tmp_path)
    assert not (tmp_path / "en_US-lessac-medium.onnx.part").exists()
