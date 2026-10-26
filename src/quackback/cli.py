"""Command line: `quackback` runs the MCP server; `download` and `try` help set it up."""

import argparse
import json
import logging
import os
import sys
from contextlib import AsyncExitStack

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
    commands.add_parser("download", help="download the voice models now, not on first use")
    commands.add_parser(
        "setup-sandbox", help="install the browser parts the simulator robot needs (Playwright)"
    )
    trial = commands.add_parser("try", help="ask a test question and print the answer")
    trial.add_argument("question", nargs="?", default="Can you hear me? Say anything you like.")
    trial.add_argument("--option", action="append", default=[], help="an answer choice")
    trial.add_argument("--keyboard", action="store_true", help="skip voice, use the dialog")
    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        serve()
    elif args.command == "download":
        download(_settings())
    elif args.command == "setup-sandbox":
        setup_sandbox()
    elif args.command == "try":
        trial_run(_settings(), args.question, args.option, args.keyboard)


def serve() -> None:
    settings = _settings()
    setup_logging(settings)
    _check_client_timeout(settings)
    from quackback.server import create_server, run_stdio

    run_stdio(create_server(settings))


def download(settings: Settings) -> None:
    from quackback.voice import Voice

    voice = Voice(settings)
    try:
        print(f"Piper voice {settings.piper_voice} ...", flush=True)
        print(f"  {voice.tts.model_path()}")
        print(f"Whisper model {settings.whisper_model} ...", flush=True)
        voice.stt.download()
    except Exception as exc:
        sys.stderr.write(f"quackback: download failed: {type(exc).__name__}: {exc}\n")
        sys.exit(1)
    print("  done")


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
        print("Loading the voice models (the first time downloads them) ...", flush=True)
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
