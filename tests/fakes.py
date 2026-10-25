"""Scripted stand-ins for the voice and keyboard channels."""

from collections.abc import Sequence

import anyio

from quackback.keyboard import KeyboardUnavailable


class FakeVoice:
    def __init__(
        self,
        heard: Sequence[str] = (),
        *,
        ready: bool = True,
        load_time: float = 0.0,
        fail: bool = False,
        listen_delay: float = 0.0,
        confirm_delay: float = 0.0,
    ) -> None:
        self.heard = list(heard)
        self.ready = ready
        self.load_time = load_time  # like models still loading: wait_ready waits for this
        self.fail = fail
        self.listen_delay = listen_delay
        self.confirm_delay = confirm_delay
        self.said: list[str] = []
        self.beeps = 0
        self.hotwords: list[list[str]] = []

    async def wait_ready(self, max_wait: float) -> bool:
        if self.load_time:
            await anyio.sleep(min(max_wait, self.load_time))
            return self.ready and self.load_time <= max_wait
        return self.ready

    async def say(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("speaker unplugged")
        if text.startswith("Got it"):
            await anyio.sleep(self.confirm_delay)
        self.said.append(text)

    async def beep(self) -> None:
        self.beeps += 1

    async def listen(self, hotwords: Sequence[str] = ()) -> str:
        self.hotwords.append(list(hotwords))
        await anyio.sleep(self.listen_delay)
        return self.heard.pop(0) if self.heard else ""


class FakeKeyboard:
    def __init__(
        self,
        answer: str | None = "typed answer",
        *,
        delay: float = 0.0,
        release: anyio.Event | None = None,
        unavailable: bool = False,
    ) -> None:
        self.answer = answer
        self.delay = delay
        self.release = release  # answer only once this is set
        self.unavailable = unavailable
        self.asked: list[tuple[str, str | None, list[str]]] = []
        self.close_after: list[float | None] = []
        self.active = 0
        self.max_active = 0

    def available(self) -> bool:
        return not self.unavailable

    async def ask(
        self,
        question: str,
        context: str | None,
        options: Sequence[str],
        close_after: float | None = None,
    ) -> str | None:
        if self.unavailable:
            raise KeyboardUnavailable("no display")
        self.asked.append((question, context, list(options)))
        self.close_after.append(close_after)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await anyio.sleep(self.delay)
            if self.release is not None:
                await self.release.wait()
        finally:
            self.active -= 1
        return self.answer
