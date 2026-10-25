"""The server over real stdio: stray stdout writes must land on stderr, never in the protocol."""

import sys
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.anyio

NOISY_SERVER = Path(__file__).with_name("noisy_server.py")


async def test_stray_stdout_does_not_corrupt_the_protocol(tmp_path: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(NOISY_SERVER)],
        cwd=str(Path(__file__).parent.parent),
        env={"PYTHONPATH": str(Path(__file__).parent.parent)},
    )
    stderr_path = tmp_path / "server-stderr.txt"
    with stderr_path.open("w", encoding="utf-8") as errlog:
        with anyio.fail_after(60):
            async with (
                stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                first = await session.call_tool("ask_human", {"question": "one"})
                second = await session.call_tool("ask_human", {"question": "two"})
    assert not first.is_error
    assert first.structured_content["answer"] == "typed: one"
    assert second.structured_content["answer"] == "typed: two"
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    assert "stray print from a library" in stderr
    assert "stray native write" in stderr
