"""Record the demo: the simulated duck walks over, asks a question, and walks home.

    uv run --extra sandbox python scripts/record_demo.py demo/

Runs the real ask_human flow (Asker + MicroduckBackend + the simulator website) with a scripted
human, so it needs no microphone and makes no sound. Writes demo/quackback-demo.mp4 (the boot
screen trimmed off) and demo/quackback-demo.gif.
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

from quackback.ask import Asker
from quackback.config import Settings
from quackback.robot.microduck import MicroduckBackend
from quackback.robot.sandbox import SandboxLink

QUESTION = "Should the cache use PostgreSQL or SQLite?"
OPTIONS = ["PostgreSQL", "SQLite"]
ANSWER = "SQLite, keep it simple."
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


async def human_eye_view(link: SandboxLink, settings: Settings) -> None:
    """Put the camera where the human stands, so the duck walks toward the viewer.

    The page's camera always follows the duck: each frame it moves the orbit target toward the
    duck and the camera by the same amount. Keeping the target on the duck and the camera in
    place every frame leaves the follow nothing to do: a person standing still, watching.
    Uses the page's debug handles (window.rl) directly: a demo nicety, not part of quackback.
    MJCF ground (x, y) is three.js (x, -z), with y up.
    """
    base = await link.pose()
    forward, left = settings.target()
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    human_x = base.x + c * forward - s * left
    human_y = base.y + s * forward + c * left
    await link._page.evaluate(
        """([hx, hy]) => {
            const rl = window.rl;
            rl.chaseCam = false;
            rl.camera.position.set(hx, 0.38, -hy);
            const eye = rl.camera.position.clone();
            const hold = () => {
                const q = rl.data.qpos;
                rl.controls.target.set(q[0], 0.1, -q[1]);
                rl.camera.position.copy(eye);
                requestAnimationFrame(hold);
            };
            hold();
        }""",
        [human_x, human_y],
    )


async def record(video_dir: Path) -> float:
    """Run one question in the simulator; return when (seconds into the video) it started."""
    settings = Settings(robot="sandbox", ask_timeout=120)
    link = SandboxLink(settings.sandbox_url, headless=True, record_dir=video_dir)
    human = ScriptedHuman()
    started = anyio.current_time()
    async with MicroduckBackend(link, settings) as robot:
        if not await robot.wait_ready(240):
            sys.exit(f"the simulator did not start: {robot.error}")
        booted = anyio.current_time() - started
        await human_eye_view(link, settings)
        await anyio.sleep(1.0)
        asker = Asker(settings, voice=human, keyboard=human, robot=robot)
        result = await asker.ask(QUESTION, None, OPTIONS)
        logging.info("ask_human returned %s", result)
        with anyio.move_on_after(90):
            await robot.go_to(0.0, 0.0)
        await anyio.sleep(1.5)
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
