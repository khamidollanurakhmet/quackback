from pathlib import Path

import pytest

from quackback.config import ConfigError, Settings, load_settings


@pytest.fixture
def no_file(tmp_path: Path) -> dict[str, str]:
    """An environment with an empty config file, so a real user config never leaks in."""
    config = tmp_path / "empty.toml"
    config.touch()
    return {"QUACKBACK_CONFIG": str(config)}


def test_defaults() -> None:
    settings = Settings()
    assert settings.ask_timeout == 300.0
    assert settings.progress_interval == 10.0
    assert settings.voice is True
    assert settings.input_device is None


def test_environment_overrides(no_file: dict[str, str]) -> None:
    env = no_file | {
        "QUACKBACK_ASK_TIMEOUT": "42",
        "QUACKBACK_VOICE": "off",
        "QUACKBACK_WHISPER_MODEL": "base.en",
        "QUACKBACK_INPUT_DEVICE": "2",
        "QUACKBACK_LOG_FILE": "~/quackback.log",
    }
    settings = load_settings(env)
    assert settings.ask_timeout == 42.0
    assert settings.voice is False
    assert settings.whisper_model == "base.en"
    assert settings.input_device == "2"
    assert settings.log_file == Path("~/quackback.log").expanduser()


def test_empty_optional_means_default(no_file: dict[str, str]) -> None:
    settings = load_settings(no_file | {"QUACKBACK_OUTPUT_DEVICE": " "})
    assert settings.output_device is None


def test_toml_file_then_environment(tmp_path: Path) -> None:
    config = tmp_path / "quackback.toml"
    config.write_text('ask_timeout = 60\nvoice = false\nlanguage = "de"\n', encoding="utf-8")
    env = {"QUACKBACK_CONFIG": str(config), "QUACKBACK_ASK_TIMEOUT": "90"}
    settings = load_settings(env)
    assert settings.ask_timeout == 90.0  # environment wins
    assert settings.voice is False
    assert settings.language == "de"


def test_unknown_toml_key_is_an_error(tmp_path: Path) -> None:
    config = tmp_path / "quackback.toml"
    config.write_text("ask_timeuot = 60\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ask_timeuot"):
        load_settings({"QUACKBACK_CONFIG": str(config)})


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing"):
        load_settings({"QUACKBACK_CONFIG": str(tmp_path / "nope.toml")})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("QUACKBACK_ASK_TIMEOUT", "soon"),
        ("QUACKBACK_ASK_TIMEOUT", "0"),
        ("QUACKBACK_VOICE", "maybe"),
        ("QUACKBACK_END_SILENCE", "-1"),
        ("QUACKBACK_ASK_TIMEOUT", "nan"),
        ("QUACKBACK_PROGRESS_INTERVAL", "inf"),
        ("QUACKBACK_END_SILENCE", "nan"),
        ("QUACKBACK_LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_values(no_file: dict[str, str], name: str, value: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(no_file | {name: value})


def test_toml_bool_for_a_number_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "quackback.toml"
    config.write_text("ask_timeout = true\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings({"QUACKBACK_CONFIG": str(config)})


def test_toml_with_a_byte_order_mark(tmp_path: Path) -> None:
    config = tmp_path / "quackback.toml"
    config.write_bytes(b"\xef\xbb\xbfask_timeout = 60\n")
    assert load_settings({"QUACKBACK_CONFIG": str(config)}).ask_timeout == 60.0


def test_utf16_toml_is_a_clear_error(tmp_path: Path) -> None:
    config = tmp_path / "quackback.toml"
    config.write_text("ask_timeout = 60\n", encoding="utf-16")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_settings({"QUACKBACK_CONFIG": str(config)})


def test_log_level_is_normalized(no_file: dict[str, str]) -> None:
    assert load_settings(no_file | {"QUACKBACK_LOG_LEVEL": "debug"}).log_level == "DEBUG"


def test_robot_settings(no_file: dict[str, str]) -> None:
    settings = load_settings(
        no_file
        | {
            "QUACKBACK_ROBOT": "sandbox",
            "QUACKBACK_ROBOT_TARGET": " 0.8 , -0.3 ",
            "QUACKBACK_MAX_SPEED": "0.2",
            "QUACKBACK_SANDBOX_HEADLESS": "yes",
        }
    )
    assert settings.robot == "sandbox"
    assert settings.target() == (0.8, -0.3)
    assert settings.max_speed == 0.2
    assert settings.max_turn is None
    assert settings.sandbox_headless is True
    assert Settings().target() == (1.5, 0.0)


def test_person_settings(no_file: dict[str, str]) -> None:
    assert Settings().find_person is True
    assert Settings().person() == (1.5, 0.0)  # where robot_target says the human is
    assert Settings(robot_target="1,1").person() == (1.0, 1.0)
    assert Settings(sandbox_person="1.2,-0.9").person() == (1.2, -0.9)
    assert Settings(sandbox_person=" None ").person() is None
    settings = load_settings(
        no_file
        | {
            "QUACKBACK_FIND_PERSON": "off",
            "QUACKBACK_PERSON_SEARCH_TIMEOUT": "12",
            "QUACKBACK_SANDBOX_PERSON_IMAGE": "~/me.png",
        }
    )
    assert settings.find_person is False
    assert settings.person_search_timeout == 12.0
    assert settings.sandbox_person_image is not None
    assert settings.sandbox_person_image.name == "me.png"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("QUACKBACK_ROBOT", "roomba"),
        ("QUACKBACK_ROBOT_TARGET", "here"),
        ("QUACKBACK_ROBOT_TARGET", "1,2,3"),
        ("QUACKBACK_ROBOT_TARGET", "nan,0"),
        ("QUACKBACK_MAX_SPEED", "0"),
        ("QUACKBACK_MAX_TURN", "nan"),
        ("QUACKBACK_SANDBOX_PERSON", "over there"),
        ("QUACKBACK_PERSON_SEARCH_TIMEOUT", "0"),
    ],
)
def test_invalid_robot_settings(no_file: dict[str, str], name: str, value: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(no_file | {name: value})
