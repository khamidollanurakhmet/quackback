"""Record the demo: the simulated duck looks for the person, walks over, asks, and walks home.

    uv run --extra sandbox python scripts/record_demo.py demo/

Runs the real ask_human flow (Asker + MicroduckBackend + person detector + the simulator
website) with a scripted human, so it needs no microphone and makes no sound. The stand-in
person stands off to the side, so the duck has to look around first; an inset shows what the
duck's camera sees and the box the detector on the PC drew. Writes demo/quackback-demo.mp4 (the
boot screen trimmed off) and demo/quackback-demo.gif.
"""

import argparse
import logging
import math
import shutil
import sys
import tempfile
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import anyio
import av
import numpy as np

from quackback.ask import Asker
from quackback.config import Settings
from quackback.robot.microduck import MicroduckBackend
from quackback.robot.sandbox import SandboxLink
from quackback.vision.person_detector import Box, PersonDetector

QUESTION = "Should the cache use PostgreSQL or SQLite?"
OPTIONS = ["PostgreSQL", "SQLite"]
ANSWER = "SQLite, keep it simple."
PERSON = "1.3,-0.7"  # forward, left of the duck's start: out of view at first
GIF_WIDTH = 640
GIF_FPS = 10
MP4_FPS = 25


class ScriptedHuman:
    """Voice and keyboard at once: "speaks" for as long as the words would take, then answers."""

    def available(self) -> bool:
        return True

    async def wait_ready(self, max_wait: float) -> bool:
        return True

    async def say(self, text: str) -> None:
        await anyio.sleep(min(6.0, len(text.split()) * 0.3))

    async def beep(self) -> None:
        await anyio.sleep(0.2)

    async def listen(self, hotwords: Sequence[str] = ()) -> str:
        await anyio.sleep(2.0)
        return ANSWER

    async def ask(self, *args: object, **kwargs: object) -> str | None:
        return ANSWER


class ShowingDetector(PersonDetector):
    """The real detector, keeping its last answer for the inset."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.last: list[Box] = []

    def detect(self, image: np.ndarray) -> list[Box]:
        self.last = super().detect(image)
        return self.last


async def watch_from_behind(link: SandboxLink, settings: Settings) -> None:
    """Hold the page camera behind and above the duck's start, the person in full view.

    The page's camera always follows the duck: each frame it moves the orbit target toward the
    duck and the camera by the same amount. Setting both every frame leaves the follow nothing
    to do. Uses the page's debug handles (window.rl) directly: a demo nicety, not part of
    quackback. MJCF ground (x, y) is three.js (x, -z), with y up.
    """
    base = await link.pose()
    forward, left = settings.person() or settings.target()
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    dx, dy = c * forward - s * left, s * forward + c * left
    length = math.hypot(dx, dy)
    camera = (base.x - 2.0 * dx / length, base.y - 2.0 * dy / length)
    target = (base.x + 0.5 * dx, base.y + 0.5 * dy)
    await link._page.evaluate(
        """([cx, cy, tx, ty]) => {
            const rl = window.rl;
            rl.chaseCam = false;
            rl.camera.fov = 50;  // wide enough for the whole person and the duck at its start
            rl.camera.updateProjectionMatrix();
            const hold = () => {
                rl.camera.position.set(cx, 1.1, -cy);
                rl.controls.target.set(tx, 0.85, -ty);
                requestAnimationFrame(hold);
            };
            hold();
        }""",
        [*camera, *target],
    )


async def show_duck_eye(link: SandboxLink, detector: ShowingDetector) -> None:
    """An inset with every picture the duck's camera takes, and the detector's box."""
    await link._page.evaluate(
        """() => {
            const inset = document.createElement("canvas");
            inset.width = 180; inset.height = 320;
            Object.assign(inset.style, {
                position: "fixed", right: "18px", bottom: "18px", zIndex: 99999,
                border: "3px solid #fff8e1", borderRadius: "10px",
                boxShadow: "0 6px 24px rgba(0,0,0,.35)", background: "#000",
            });
            document.body.appendChild(inset);
            const g = inset.getContext("2d");
            const qb = window.__quackback, eye = qb.eye;
            qb.eye = () => {
                const shot = eye();
                const img = new Image();
                img.onload = () => {
                    g.drawImage(img, 0, 0, 180, 320);
                    const box = window.__demoBox;
                    g.font = "600 13px system-ui, sans-serif";
                    if (box) {
                        g.strokeStyle = "#ffd54f"; g.lineWidth = 3;
                        g.strokeRect(box[0] * 180, box[1] * 320,
                                     (box[2] - box[0]) * 180, (box[3] - box[1]) * 320);
                        g.fillStyle = "#ffd54f";
                        const label = `person ${box[4].toFixed(2)}`;
                        g.fillText(label, box[0] * 180 + 4, box[1] * 320 + 16);
                    }
                    g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(0, 296, 180, 24);
                    g.fillStyle = "#fff8e1"; g.fillText("duck's camera", 8, 313);
                };
                img.src = "data:image/jpeg;base64," + shot.jpeg;
                return shot;
            };
        }"""
    )
    while True:
        boxes = detector.last
        box = None
        if boxes:
            b = boxes[0]
            box = [b.left, b.top, b.right, b.bottom, b.score]
        await link._page.evaluate("(box) => { window.__demoBox = box; }", box)
        await anyio.sleep(0.1)


async def record(video_dir: Path) -> float:
    """Run one question in the simulator; return when (seconds into the video) it started."""
    settings = Settings(robot="sandbox", ask_timeout=180, sandbox_person=PERSON)
    link = SandboxLink(
        settings.sandbox_url, headless=True, record_dir=video_dir, person=settings.person()
    )
    human = ScriptedHuman()
    detector = ShowingDetector(settings)
    started = anyio.current_time()
    async with MicroduckBackend(link, settings, detector) as robot:
        if not await robot.wait_ready(240):
            sys.exit(f"the simulator did not start: {robot.error}")
        if not robot.can_find_person:
            sys.exit("no person detector: run with the sandbox extra")
        booted = anyio.current_time() - started
        await watch_from_behind(link, settings)
        async with anyio.create_task_group() as tg:
            tg.start_soon(show_duck_eye, link, detector)
            await anyio.sleep(1.0)
            asker = Asker(settings, voice=human, keyboard=human, robot=robot)
            result = await asker.ask(QUESTION, None, OPTIONS)
            logging.info("ask_human returned %s", result)
            with anyio.move_on_after(90):
                await robot.go_to(0.0, 0.0)
            await anyio.sleep(1.5)
            tg.cancel_scope.cancel()
    return booted


def transcode(webm: Path, mp4: Path, gif: Path, start_s: float) -> None:
    """Cut the boot screen, then write an H.264 mp4 and a small GIF, each frame renumbered."""
    with av.open(str(webm)) as source:
        stream = source.streams.video[0]
        with av.open(str(mp4), "w") as out_mp4, av.open(str(gif), "w") as out_gif:
            h264 = out_mp4.add_stream("libx264", rate=MP4_FPS)
            h264.width, h264.height, h264.pix_fmt = stream.width, stream.height, "yuv420p"
            h264.options = {"crf": "23", "preset": "medium"}
            palette = out_gif.add_stream("gif", rate=GIF_FPS)
            palette.width = GIF_WIDTH
            palette.height = round(stream.height * GIF_WIDTH / stream.width / 2) * 2
            palette.pix_fmt = "rgb8"
            outputs = (
                (out_mp4, h264, MP4_FPS, "yuv420p", None),
                (out_gif, palette, GIF_FPS, "rgb8", (palette.width, palette.height)),
            )
            written = [0, 0]
            for frame in source.decode(stream):
                t = float(frame.time or 0.0) - start_s
                if t < 0:
                    continue
                for i, (container, out, fps, pix_fmt, size) in enumerate(outputs):
                    # Each output frame takes the source frame showing at its own tick.
                    while written[i] <= t * fps:
                        width, height = size or (stream.width, stream.height)
                        copy = frame.reformat(width, height, format=pix_fmt)
                        copy.pts = written[i]
                        # Explicit: the stream's own time base is unset until the first mux.
                        copy.time_base = Fraction(1, fps)
                        written[i] += 1
                        for packet in out.encode(copy):
                            container.mux(packet)
            for container, out, *_ in outputs:
                for packet in out.encode():
                    container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory for the mp4 and gif")
    parser.add_argument(
        "--transcode-only", type=float, metavar="START_S",
        help="re-encode an earlier quackback-demo-raw.webm from START_S seconds",
    )  # fmt: skip
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "quackback-demo-raw.webm"
    start_s = args.transcode_only
    if start_s is None:
        with tempfile.TemporaryDirectory() as tmp:
            start_s = anyio.run(record, Path(tmp))
            shutil.copy(next(Path(tmp).glob("*.webm")), raw)
        logging.info("the simulator was up %.1f s into the recording", start_s)
    transcode(raw, args.out / "quackback-demo.mp4", args.out / "quackback-demo.gif", start_s)
    logging.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
