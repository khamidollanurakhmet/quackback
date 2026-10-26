"""The person detector: its model download (against a local server), and the real model on the
simulator's stand-in person.

The real-model test is opt-in (needs the vision extra and a 4.6 MB download):
QUACKBACK_TEST_MODELS=1 uv run --extra vision pytest tests/test_vision.py
"""

import hashlib
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from quackback.config import Settings, default_cache_dir
from quackback.vision import person_detector
from quackback.vision.person_detector import MODEL_FILE, PersonDetector

MODEL = b"not really a model" * 1000


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(MODEL)))
        self.end_headers()
        self.wfile.write(MODEL)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/model.tflite"
    httpd.shutdown()


def test_model_is_downloaded_once_and_checked(
    server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(person_detector, "MODEL_URL", server)
    monkeypatch.setattr(person_detector, "MODEL_SHA256", hashlib.sha256(MODEL).hexdigest())
    detector = PersonDetector(Settings(models_dir=tmp_path))
    path = detector.model_path()
    assert path == tmp_path / "vision" / MODEL_FILE
    assert path.read_bytes() == MODEL
    monkeypatch.setattr(person_detector, "MODEL_URL", "http://127.0.0.1:9/unreachable")
    assert detector.model_path() == path  # already there: no download


def test_a_model_that_is_not_the_pinned_one_is_thrown_away(
    server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(person_detector, "MODEL_URL", server)
    detector = PersonDetector(Settings(models_dir=tmp_path))
    with pytest.raises(RuntimeError, match="not the expected one"):
        detector.model_path()
    assert not (tmp_path / "vision" / MODEL_FILE).exists()


@pytest.mark.models
@pytest.mark.skipif(
    os.environ.get("QUACKBACK_TEST_MODELS") != "1", reason="set QUACKBACK_TEST_MODELS=1"
)
def test_the_real_detector_finds_the_stand_in_person() -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    picture = Path(person_detector.__file__).parents[1] / "robot" / "assets" / "stand_in_person.png"
    cutout = cv2.imread(str(picture), cv2.IMREAD_UNCHANGED)
    rgb, alpha = cutout[:, :, 2::-1], cutout[:, :, 3:] / 255.0
    # On a grey floor, a little left of the middle of a portrait picture like the duck's.
    frame = np.full((640, 360, 3), 128, np.uint8)
    height = 500
    width = round(height * rgb.shape[1] / rgb.shape[0])
    small = cv2.resize(rgb, (width, height))
    mask = cv2.resize(alpha, (width, height))[:, :, None]
    patch = frame[70 : 70 + height, 60 : 60 + width]
    frame[70 : 70 + height, 60 : 60 + width] = (small * mask + patch * (1 - mask)).astype(np.uint8)
    detector = PersonDetector(Settings(models_dir=default_cache_dir()))
    detector.load()
    (best, *_) = detector.detect(frame)
    assert best.score > 0.6
    assert best.left == pytest.approx(60 / 360, abs=0.05)
    assert best.right == pytest.approx((60 + width) / 360, abs=0.05)
    assert detector.detect(np.full((640, 360, 3), 128, np.uint8)) == []  # nobody on a bare floor
