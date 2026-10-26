"""Settings: defaults, overridden by a TOML file, overridden by QUACKBACK_* environment variables.

Every field `foo_bar` is the TOML key `foo_bar` and the environment variable `QUACKBACK_FOO_BAR`.
The TOML file is `QUACKBACK_CONFIG` if set, else `config.toml` in the user config directory
(see `default_config_path`).
"""

import math
import os
import sys
import tomllib
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin

ENV_PREFIX = "QUACKBACK_"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """A setting has an invalid value or an unknown name."""


@dataclass(frozen=True)
class Settings:
    # How long ask_human waits for an answer before giving up, seconds.
    ask_timeout: float = 300.0
    # How often a progress notification is sent while waiting, seconds.
    progress_interval: float = 10.0
    # False: skip speech entirely and go straight to the keyboard dialog.
    voice: bool = True
    # Whisper language code; empty string lets Whisper detect it.
    language: str = "en"
    # A faster-whisper model name ("base.en", "small.en", "large-v3-turbo", ...) or a local path.
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # A Piper voice name ("en_US-lessac-medium") or a path to a local .onnx voice.
    piper_voice: str = "en_US-lessac-medium"
    # Where downloaded Piper voices are kept; empty means the user cache directory.
    models_dir: Path | None = None
    # sounddevice device name or index; empty means the system default.
    input_device: str | None = None
    output_device: str | None = None
    # Give up on an answer if no speech starts within this many seconds of the beep.
    listen_timeout: float = 8.0
    # Silence that ends an answer, seconds.
    end_silence: float = 1.2
    # The longest answer that is recorded, seconds.
    max_answer: float = 30.0
    log_level: str = "INFO"
    # Also log to this file (logs always go to stderr; stdout is the MCP transport).
    log_file: Path | None = None

    # The robot that walks over to ask: "none", or "sandbox" (the Microduck simulator website).
    robot: str = "none"
    # Where the human is, in metres from the robot's start: "forward,left". The robot stops
    # 0.4 m short of it.
    robot_target: str = "1.5,0.0"
    # Speed caps; empty means the robot's own maximum (the sandbox walks only at 0.25 m/s).
    max_speed: float | None = None
    max_turn: float | None = None
    # The simulator page, and the browser it runs in: a Playwright channel ("msedge",
    # "chrome") or empty for Edge on Windows, Chrome elsewhere, else Playwright's Chromium.
    sandbox_url: str = "https://pollen-robotics-microduck-simulator.hf.space/?boot=1"
    sandbox_browser: str = ""
    sandbox_headless: bool = False
    # Record the simulator window as video into this directory.
    sandbox_record_dir: Path | None = None

    def __post_init__(self) -> None:
        for name in ("ask_timeout", "progress_interval", "listen_timeout", "end_silence"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ConfigError(f"{name} must be a positive number of seconds, got {value}")
        if self.max_answer <= self.end_silence:
            raise ConfigError("max_answer must be longer than end_silence")
        if self.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ConfigError(f"log_level must be DEBUG, INFO, WARNING or ERROR: {self.log_level}")
        object.__setattr__(self, "log_level", self.log_level.upper())
        if self.robot not in ROBOTS:
            raise ConfigError(f"robot must be one of {', '.join(ROBOTS)}: {self.robot}")
        self.target()  # validates robot_target
        for name in ("max_speed", "max_turn"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ConfigError(f"{name} must be a positive number, got {value}")

    def target(self) -> tuple[float, float]:
        """robot_target as (forward, left) metres from the robot's start."""
        try:
            forward, left = (float(part) for part in self.robot_target.split(","))
        except ValueError:
            raise ConfigError(
                f'robot_target must be "forward,left" in metres: {self.robot_target!r}'
            ) from None
        if not (math.isfinite(forward) and math.isfinite(left)):
            raise ConfigError(f"robot_target must be finite: {self.robot_target!r}")
        return forward, left


ROBOTS = ("none", "sandbox")


def default_config_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "quackback" / "config.toml"


def default_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "quackback"


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    explicit = env.get(ENV_PREFIX + "CONFIG")
    path = Path(explicit) if explicit else default_config_path()
    values: dict[str, Any] = {}
    if path.is_file():
        values.update(_read_toml(path))
    elif explicit:
        raise ConfigError(f"{ENV_PREFIX}CONFIG points to a missing file: {path}")
    for f in fields(Settings):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is not None:
            values[f.name] = raw
    return Settings(**{name: _coerce(name, value) for name, value in values.items()})


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        # utf-8-sig: Windows editors and PowerShell 5.1 like to write a byte order mark.
        data = tomllib.loads(path.read_bytes().decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path}: not UTF-8 text, save it as UTF-8 ({exc})") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read it ({exc})") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{path}: unknown settings {', '.join(unknown)}")
    return data


def _coerce(name: str, value: Any) -> Any:
    target = {f.name: f.type for f in fields(Settings)}[name]
    optional = get_origin(target) in (Union, types.UnionType) and type(None) in get_args(target)
    if optional:
        target = next(arg for arg in get_args(target) if arg is not type(None))
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
    try:
        if target is bool:
            return _to_bool(value)
        if target is float:
            if isinstance(value, bool):
                raise ValueError("expected a number")
            return float(value)
        if target is Path:
            return Path(str(value)).expanduser()
        return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} = {value!r}: {exc}") from exc


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError("expected true or false")
