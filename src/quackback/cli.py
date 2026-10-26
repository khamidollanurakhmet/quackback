"""Command line: `quackback` runs the MCP server; `download` and `try` help set it up."""

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from pathlib import Path

import anyio

from quackback import __version__
from quackback.config import ConfigError, Settings, load_settings

log = logging.getLogger("quackback")

# The client needs this much slack beyond quackback's own timeout to receive the timeout result.
CLIENT_TIMEOUT_MARGIN_MS = 5_000
# `quackback try` waits this long for the simulator to boot, and for the duck to walk home.
TRY_ROBOT_READY_S = 240.0
TRY_HOME_S = 90.0


def main(argv: list[str] | None = None) -> None:
    # Before anything else: in `serve`, stdout is about to become the MCP transport, and a
    # non-UTF-8 console encoding must not turn a stray character into a crash in any command.
    _utf8_std_streams()
    parser = argparse.ArgumentParser(
        prog="quackback",
        description="An MCP server whose ask_human tool asks you questions out loud.",
    )
    parser.add_argument("--version", action="version", version=f"quackback {__version__}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the MCP server on stdio (the default)")
    commands.add_parser("download", help="download the models now, not on first use")
    commands.add_parser(
        "setup-sandbox", help="install the browser parts the simulator robot needs (Playwright)"
    )
    trial = commands.add_parser("try", help="ask a test question and print the answer")
    trial.add_argument("question", nargs="?", default="Can you hear me? Say anything you like.")
    trial.add_argument("--option", action="append", default=[], help="an answer choice")
    trial.add_argument("--keyboard", action="store_true", help="skip voice, use the dialog")
    args = parser.parse_args(argv)

    try:
        if args.command in (None, "serve"):
            serve()
        elif args.command == "download":
            download(_settings())
        elif args.command == "setup-sandbox":
            setup_sandbox()
        elif args.command == "try":
            trial_run(_settings(), args.question, args.option, args.keyboard)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted. A download picks up where it stopped next time.\n")
        sys.exit(130)


def serve() -> None:
    settings = _settings()
    setup_logging(settings)
    _check_client_timeout(settings)
    from quackback.server import create_server, run_stdio

    run_stdio(create_server(settings))


def download(settings: Settings) -> None:
    try:
        prepare_models(settings)
    except Exception as exc:
        sys.stderr.write(f"quackback: download failed: {type(exc).__name__}: {exc}\n")
        sys.exit(1)
    print("done")


def prepare_models(settings: Settings) -> None:
    """Make sure the models are on disk, showing the downloads as they happen."""
    from quackback.vision.person_detector import PersonDetector, vision_installed
    from quackback.voice import Voice

    voice = Voice(settings)
    print(f"Piper voice {settings.piper_voice}", flush=True)
    path = voice.tts.model_path(TerminalProgress())
    print(f"  ready: {path}", flush=True)
    print(f"Whisper model {settings.whisper_model}", flush=True)
    if voice.stt.is_downloaded():
        print("  ready (already downloaded)", flush=True)
    else:
        print("  downloading, a few hundred MB the first time ...", flush=True)
        with growth_counter(voice.stt.cache_dir()):
            voice.stt.download()
        print("  ready", flush=True)
    if vision_installed():
        # Optional: without it the robot walks to robot_target instead of looking for you.
        print("Person detector (EfficientDet-Lite0)", flush=True)
        try:
            path = PersonDetector(settings).model_path(TerminalProgress())
        except Exception as exc:
            print(f"  not downloaded ({type(exc).__name__}: {exc}); it is tried again on first use")
        else:
            print(f"  ready: {path}", flush=True)


def format_progress(name: str, done: int, total: int | None) -> str:
    if total:
        return f"  {name}  {done / 1e6:.1f} / {total / 1e6:.1f} MB  {done * 100 // total}%"
    return f"  {name}  {done / 1e6:.1f} MB"


class TerminalProgress:
    """A download's progress on one terminal line, redrawn in place a few times a second."""

    def __init__(self, interval: float = 0.2) -> None:
        self._interval = interval
        self._drawn_at = -math.inf

    def __call__(self, name: str, done: int, total: int | None) -> None:
        finished = total is not None and done >= total
        now = time.monotonic()
        if not finished and now - self._drawn_at < self._interval:
            return
        self._drawn_at = now
        sys.stdout.write("\r" + format_progress(name, done, total).ljust(64))
        if finished:
            sys.stdout.write("\n")
        sys.stdout.flush()


@contextmanager
def growth_counter(folder: Path | None) -> Iterator[None]:
    """Show how much has arrived in `folder/blobs` while the block runs."""
    stop = threading.Event()
    started = time.monotonic()

    def count() -> None:
        while not stop.wait(0.5):
            blobs = folder / "blobs" if folder else None
            size = 0
            if blobs is not None and blobs.is_dir():
                size = sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())
            elapsed = time.monotonic() - started
            sys.stdout.write(f"\r  {size / 1e6:.0f} MB so far, {elapsed:.0f} s".ljust(64))
            sys.stdout.flush()

    thread = threading.Thread(target=count, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        sys.stdout.write("\n")


def setup_sandbox() -> None:
    """Playwright's Chromium (when there is no Edge or Chrome) and its ffmpeg (for recording).

    Runs Playwright through this interpreter: after `uv tool install`, the `playwright` command
    itself is not on PATH.
    """
    import subprocess

    try:
        import playwright  # noqa: F401
    except ImportError:
        sys.stderr.write('quackback: install the extra first: "quackback[sandbox]"\n')
        sys.exit(1)
    command = [sys.executable, "-m", "playwright", "install", "chromium", "ffmpeg"]
    print("Running:", " ".join(command), flush=True)
    sys.exit(subprocess.run(command, check=False).returncode)


def trial_run(settings: Settings, question: str, options: list[str], keyboard: bool) -> None:
    from quackback.ask import Asker
    from quackback.keyboard import KeyboardPrompt
    from quackback.robot.base import RobotError
    from quackback.server import open_robot
    from quackback.voice import Voice

    setup_logging(settings)
    voice = None
    if not keyboard and settings.voice:
        try:
            prepare_models(settings)
        except Exception as exc:  # load() below fails too, and reports it
            print(f"Could not download the voice models: {type(exc).__name__}: {exc}")
        print("Loading the voice models ...", flush=True)
        voice = Voice(settings)
        voice.load()
        if voice.error:
            print(f"Voice unavailable, falling back to the keyboard: {voice.error}")

    async def run() -> None:
        async with AsyncExitStack() as stack:
            robot = await open_robot(settings, stack)
            if robot is not None:
                print("Starting the simulator (about half a minute) ...", flush=True)
                if not await robot.wait_ready(TRY_ROBOT_READY_S):
                    print(f"Robot unavailable: {getattr(robot, 'error', None) or 'timed out'}")
            asker = Asker(settings, voice=voice, keyboard=KeyboardPrompt(), robot=robot)
            result = await asker.ask(question, None, options)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if robot is not None and await robot.wait_ready(0):
                print("Walking home ...", flush=True)
                with anyio.move_on_after(TRY_HOME_S):
                    try:
                        await robot.go_to(0.0, 0.0)
                    except RobotError as exc:
                        print(f"Did not make it home: {exc}")

    anyio.run(run)


def setup_logging(settings: Settings) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if settings.log_file:
        try:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
        except OSError as exc:
            sys.stderr.write(f"quackback: cannot open the log file {settings.log_file}: {exc}\n")
            sys.exit(2)
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("mcp", "httpx", "faster_whisper", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.root.level))


def _settings() -> Settings:
    try:
        return load_settings()
    except ConfigError as exc:
        sys.stderr.write(f"quackback: {exc}\n")
        sys.exit(2)


def check_client_timeout(settings: Settings, env: dict[str, str] | None = None) -> str | None:
    """A warning when the MCP client would give up on ask_human before quackback does."""
    raw = (os.environ if env is None else env).get("MCP_TOOL_TIMEOUT", "").strip()
    needed = int(settings.ask_timeout * 1000) + CLIENT_TIMEOUT_MARGIN_MS
    if raw.isdigit() and int(raw) < needed:
        return (
            f"MCP_TOOL_TIMEOUT is {raw} ms but QUACKBACK_ASK_TIMEOUT is "
            f"{settings.ask_timeout:g} s: the client would cancel ask_human before it can "
            f"return. Set MCP_TOOL_TIMEOUT to at least {needed}."
        )
    return None


def _check_client_timeout(settings: Settings) -> None:
    warning = check_client_timeout(settings)
    if warning:
        log.warning(warning)


def _utf8_std_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")
