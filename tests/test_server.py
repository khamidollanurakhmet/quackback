import json

import anyio
import pytest
from mcp.client import Client

from quackback.ask import Asker
from quackback.config import Settings
from quackback.server import create_server
from tests.fakes import FakeKeyboard, FakeVoice

pytestmark = pytest.mark.anyio


def _server(voice: FakeVoice | None = None, keyboard: FakeKeyboard | None = None, **kw: object):
    settings = Settings(**kw)  # type: ignore[arg-type]
    asker = Asker(settings, voice=voice, keyboard=keyboard or FakeKeyboard())
    return create_server(settings, asker=asker)


async def test_lists_the_tools_with_a_safe_description() -> None:
    async with Client(_server(ask_timeout=120), mode="legacy") as client:
        tools = (await client.list_tools()).tools
    assert [tool.name for tool in tools] == ["ask_human", "stop_robot"]
    tool = tools[0]
    assert "120 seconds" in tool.description
    assert "never authorization" in tool.description
    assert tool.input_schema["required"] == ["question"]
    assert set(tool.input_schema["properties"]) == {"question", "context", "options"}
    assert set(tool.output_schema["required"]) == {"answer", "source", "duration_s"}


async def test_call_returns_the_answer() -> None:
    voice = FakeVoice(["the second one"])
    async with Client(_server(voice), mode="legacy") as client:
        result = await client.call_tool(
            "ask_human",
            {"question": "Which?", "context": "Pick a DB.", "options": ["Postgres", "SQLite"]},
        )
    assert not result.is_error
    answer = result.structured_content
    assert answer["answer"] == "SQLite"
    assert answer["source"] == "voice"
    assert answer["transcript"] == "the second one"
    assert json.loads(result.content[0].text)["answer"] == "SQLite"
    assert "Pick a DB." in voice.said[0]


async def test_null_context_and_json_string_options() -> None:
    voice = FakeVoice(["b"])
    async with Client(_server(voice), mode="legacy") as client:
        result = await client.call_tool(
            "ask_human", {"question": "Which?", "context": None, "options": '["a", "b"]'}
        )
    assert not result.is_error
    assert result.structured_content["answer"] == "b"
    assert voice.said[0] == "Your coding agent has a question. Which? Option 1: a. Option 2: b."


async def test_progress_notifications() -> None:
    seen: list[tuple[float, float | None, str | None]] = []
    release = anyio.Event()

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))
        if len(seen) >= 2:
            release.set()

    keyboard = FakeKeyboard(release=release)
    server = _server(keyboard=keyboard, progress_interval=0.05, ask_timeout=90)
    with anyio.fail_after(10):
        async with Client(server, mode="legacy") as client:
            result = await client.call_tool(
                "ask_human", {"question": "Q?"}, progress_callback=on_progress
            )
    assert not result.is_error
    assert len(seen) >= 2
    assert all(total == 90 for _, total, _ in seen)


async def test_unreachable_human_is_a_tool_error() -> None:
    async with Client(_server(keyboard=FakeKeyboard(unavailable=True)), mode="legacy") as client:
        result = await client.call_tool("ask_human", {"question": "Q?"})
    assert result.is_error
    assert "cannot reach the human" in result.content[0].text


async def test_timeout_is_a_normal_result() -> None:
    server = _server(keyboard=FakeKeyboard(delay=30), ask_timeout=0.3)
    async with Client(server, mode="legacy") as client:
        result = await client.call_tool("ask_human", {"question": "Q?"})
    assert not result.is_error
    assert result.structured_content["source"] == "timeout"


async def test_missing_question_is_rejected() -> None:
    async with Client(_server(), mode="legacy") as client:
        result = await client.call_tool("ask_human", {"context": "no question"})
    assert result.is_error


async def test_stop_robot_without_a_robot() -> None:
    async with Client(_server(), mode="legacy") as client:
        result = await client.call_tool("stop_robot", {})
    assert not result.is_error
    assert result.structured_content == {"result": "There is no robot."}


async def test_stop_robot_stops_it() -> None:
    from quackback.robot.mock import MockBackend

    settings = Settings()
    robot = MockBackend()
    server = create_server(settings, asker=Asker(settings, None, FakeKeyboard(), robot=robot))
    async with Client(server, mode="legacy") as client:
        result = await client.call_tool("stop_robot", {})
    assert result.structured_content == {"result": "The robot stopped."}
    assert robot.calls == ["stop"]
