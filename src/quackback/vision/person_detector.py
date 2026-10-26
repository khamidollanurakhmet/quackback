"""Finding people in the duck's camera pictures with MediaPipe's object detector.

The model is EfficientDet-Lite0 (int8, 4.6 MB), downloaded once into the models directory and
checked against a pinned hash. It runs on the CPU in 20-60 ms a picture. MediaPipe is the
optional `vision` extra, imported only when the detector loads (that alone takes seconds).

MediaPipe sends Google usage metrics (not pictures) when a detector is closed, inside its native
close call, with no switch to turn that off. quackback never closes one: a detector lives as
long as the process, one that breaks is dropped without that call, and at exit every detector
is detached first, so its finalizer has nothing left to close.
"""

import atexit
import hashlib
import importlib.util
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from quackback.config import Settings, default_cache_dir
from quackback.voice.tts import LogProgress, Progress, download

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)
MODEL_SHA256 = "0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb"
MODEL_FILE = "efficientdet_lite0_int8.tflite"
# Never 0: MediaPipe then decodes every anchor box, about 19 times slower.
SCORE_THRESHOLD = 0.35
MAX_RESULTS = 3


@dataclass(frozen=True)
class Box:
    """A person in a picture: the edges as fractions of its width and height, from the top left."""

    left: float
    top: float
    right: float
    bottom: float
    score: float


class Detector(Protocol):
    def load(self) -> None:
        """Get ready (download and load the model). Blocking; raises if it cannot."""
        ...

    def detect(self, image: NDArray[np.uint8]) -> list[Box]:
        """The people in an RGB picture, most certain first. Blocking."""
        ...


def vision_installed() -> bool:
    return importlib.util.find_spec("mediapipe") is not None


class PersonDetector:
    def __init__(self, settings: Settings) -> None:
        self._dir = (settings.models_dir or default_cache_dir()) / "vision"
        self._lock = threading.Lock()  # one picture at a time: a detector is not reentrant
        self._detector: Any = None

    def model_path(self, progress: Progress | None = None) -> Path:
        """The model on disk, downloading it if needed. Blocking."""
        path = self._dir / MODEL_FILE
        if path.is_file() and _sha256(path) == MODEL_SHA256:
            path.with_name(path.name + ".part").unlink(missing_ok=True)
            return path
        self._dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading the person detector model")
        download(MODEL_URL, path, progress)
        if _sha256(path) != MODEL_SHA256:
            path.unlink()
            raise RuntimeError(f"the person detector model at {MODEL_URL} is not the expected one")
        return path

    def load(self) -> None:
        with self._lock:
            if self._detector is None:
                self._detector = self._create()

    def detect(self, image: NDArray[np.uint8]) -> list[Box]:
        import mediapipe as mp

        height, width = image.shape[:2]
        picture = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        with self._lock:
            if self._detector is None:
                self._detector = self._create()
            try:
                result = self._detector.detect(picture)
            except Exception:
                # After an error in its graph a detector stays broken: the next picture gets a
                # new one.
                _abandon(self._detector)
                self._detector = None
                raise
        boxes = []
        for detection in result.detections:
            box = detection.bounding_box
            boxes.append(
                Box(
                    left=box.origin_x / width,
                    top=box.origin_y / height,
                    right=(box.origin_x + box.width) / width,
                    bottom=(box.origin_y + box.height) / height,
                    score=detection.categories[0].score,
                )
            )
        return sorted(boxes, key=lambda b: b.score, reverse=True)

    def _create(self) -> Any:
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.ObjectDetectorOptions(
            # The model's bytes, not its path: MediaPipe cannot open a path with non-ASCII
            # characters on Windows.
            base_options=BaseOptions(
                model_asset_buffer=self.model_path(LogProgress()).read_bytes()
            ),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=SCORE_THRESHOLD,
            category_allowlist=["person"],
            max_results=MAX_RESULTS,
        )
        detector = vision.ObjectDetector.create_from_options(options)
        _never_closed.append(detector)
        log.info("person detector ready")
        return detector


# Every detector ever made: kept referenced, so none is finalized (and closed) before exit.
_never_closed: list[Any] = []


def _abandon(detector: Any) -> None:
    """Drop a detector without MediaPipe's close call, stopping its worker threads only."""
    detector._handle = None
    for name in ("_dispatcher", "_lib"):
        try:
            getattr(detector, name).close()
        except Exception:
            log.debug("could not stop a detector's %s", name, exc_info=True)


@atexit.register
def _detach_all() -> None:
    for detector in _never_closed:
        detector._handle = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
