"""The ask_human flow: speak the question, listen, fall back to the keyboard, give up on time.

One question at a time (there is one human). The timeout covers the whole call, waiting for an
earlier question included. Voice failures never escape: they log and hand over to the keyboard.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol

import anyio
from typing_extensions import TypedDict

from quackback.config import Settings
from quackback.keyboard import KeyboardUnavailable
from quackback.options import match_option
from quackback.robot.base import HUMAN_STOP_DISTANCE, RobotBackend, RobotError

log = logging.getLogger(__name__)

NO_ANSWER = "The human did not answer. Make a reasonable decision and note it."
# How long a question waits for the voice models to finish loading before using the keyboard;
# never more than a third of the ask timeout, so the keyboard still gets its turn.
VOICE_READY_WAIT_S = 15.0
# "Got it." is spoken after the answer is safely recorded, and may run past the ask timeout.
CONFIRM_TIMEOUT_S = 5.0
# Like the voice: how long a question waits for the robot to come up, at most.
ROBOT_READY_WAIT_S = 15.0

Source = Literal["voice", "keyboard", "timeout"]
Progress = Callable[[float, float, str], Awaitable[None]]


class AskResult(TypedDict):
    answer: str
    source: Source
    duration_s: float
    transcript: NotRequired[str]  # the raw words, when options were given


class AskFailed(RuntimeError):
    """No channel could reach the human: voice failed and the keyboard dialog cannot open."""


class VoiceChannel(Protocol):
    async def wait_ready(self, max_wait: float) -> bool: ...
    async def say(self, text: str) -> None: ...
    async def beep(self) -> None: ...
    async def listen(self, hotwords: Sequence[str] = ()) -> str: ...


class KeyboardChannel(Protocol):
    def available(self) -> bool: ...
    async def ask(
        self,
        question: str,
        context: str | None,
        options: Sequence[str],
        close_after: float | None = None,
    ) -> str | None: ...


@dataclass
class _Reply:
    text: str
    source: Source
    at: float  # when the answer arrived (anyio clock)


def spoken_question(question: str, context: str | None, options: Sequence[str]) -> str:
    parts = ["Your coding agent has a question."]
    if context:
        parts.append(_sentence(context))
    parts.append(_sentence(question))
    parts.extend(f"Option {i}: {_sentence(option)}" for i, option in enumerate(options, 1))
    return " ".join(parts)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


class Asker:
    def __init__(
        self,
        settings: Settings,
        voice: VoiceChannel | None,
        keyboard: KeyboardChannel,
        robot: RobotBackend | None = None,
    ) -> None:
        self._settings = settings
        self._voice = voice
        self._keyboard = keyboard
        self._robot = robot
        self._lock = anyio.Lock()

    async def ask(
        self,
        question: str,
        context: str | None = None,
        options: Sequence[str] | None = None,
        progress: Progress | None = None,
    ) -> AskResult:
        """Ask and wait. Raises AskFailed only if the human cannot be reached at all."""
        options = [option for option in (options or []) if option.strip()]
        started = anyio.current_time()
        deadline = started + self._settings.ask_timeout
        reply: _Reply | None = None
        failure: AskFailed | None = None
        # Errors are kept and raised after the task group, which would wrap them in a group.
        async with anyio.create_task_group() as tg:
            if progress is not None:
                tg.start_soon(self._report_progress, progress, started)
            try:
                reply = await self._ask_in_turn(question, context, options, deadline)
            except AskFailed as exc:
                failure = exc
            finally:
                tg.cancel_scope.cancel()
        if failure is not None:
            raise failure
        if reply is None:
            duration = round(anyio.current_time() - started, 1)
            log.info("no answer after %.0f s", duration)
            return {"answer": NO_ANSWER, "source": "timeout", "duration_s": duration}
        duration = round(reply.at - started, 1)
        result: AskResult = {"answer": reply.text, "source": reply.source, "duration_s": duration}
        if options:
            result["transcript"] = reply.text
            result["answer"] = match_option(reply.text, options) or reply.text
        return result

    async def _ask_in_turn(
        self, question: str, context: str | None, options: list[str], deadline: float
    ) -> _Reply | None:
        """Wait for any earlier question, then ask; both within the deadline."""
        acquired = False
        with anyio.move_on_after(_remaining(deadline)):
            await self._lock.acquire()
            acquired = True
        if not acquired:
            return None
        reply = None
        try:
            try:
                with anyio.move_on_after(_remaining(deadline)):
                    await self._approach(question)
                    reply = await self._ask(question, context, options, deadline)
                if reply is not None and reply.source == "voice" and self._voice is not None:
                    with anyio.move_on_after(CONFIRM_TIMEOUT_S):
                        await self._confirm(self._voice, reply.text, options)
                return reply
            finally:
                # Before the next question may start, so its walk supersedes this trip home.
                await self._dismiss_robot(reply, options)
        finally:
            self._lock.release()

    async def stop_robot(self) -> bool:
        """Emergency stop. False if there is no robot."""
        if self._robot is None:
            return False
        await self._robot.emergency_stop()
        return True

    async def _approach(self, question: str) -> None:
        """Walk the robot to the human and quack. Without a robot, or if it fails, carry on."""
        robot = self._robot
        if robot is None:
            return
        try:
            # A new question lifts an earlier emergency stop; a stop from here on holds for it.
            robot.resume()
            budget = self._settings.ask_timeout / 3
            if not await robot.wait_ready(min(ROBOT_READY_WAIT_S, budget)):
                log.info("the robot is not ready, asking without it")
                return
            await robot.show_text(question)
            # The walk gets a third of the time at most: the question must still be asked.
            arrived = False
            with anyio.move_on_after(budget) as walk:
                if robot.can_find_person:
                    arrived = await robot.approach_person(self._settings.person_search_timeout)
                else:
                    await robot.go_to(*self._settings.target(), stop_short=HUMAN_STOP_DISTANCE)
                    arrived = True
            if arrived:
                await robot.cue()
            elif walk.cancelled_caught:
                log.warning("the robot is too slow, asking from where it is")
            else:
                log.info("the robot found nobody, asking on this computer")
        except RobotError as exc:
            log.warning("robot failed, asking without it: %s", exc)
        except Exception:  # a robot bug must not take the question down with it
            log.exception("robot failed unexpectedly, asking without it")

    async def _dismiss_robot(self, reply: _Reply | None, options: list[str]) -> None:
        """Show the answer (or clear the question) and send the robot home; on time, out of
        time, or after a failure."""
        robot = self._robot
        if robot is None:
            return
        answer = None
        if reply is not None:
            answer = (
                f"Got it: {(match_option(reply.text, options) if options else None) or reply.text}"
            )
        with anyio.CancelScope(shield=True), anyio.move_on_after(2):
            try:
                await robot.show_text(answer)
            except Exception as exc:
                log.debug("could not update the robot's caption: %s", exc)
        try:
            robot.send_home()
        except Exception:
            log.exception("could not send the robot home")

    async def _ask(
        self, question: str, context: str | None, options: list[str], deadline: float
    ) -> _Reply | None:
        """A reply, or None if the question went unanswered (dismissed, or no screen either)."""
        heard, spoken = await self._by_voice(question, context, options)
        if heard:
            return _Reply(heard, "voice", anyio.current_time())
        try:
            text = await self._keyboard.ask(
                question, context, options, close_after=_remaining(deadline)
            )
        except KeyboardUnavailable as exc:
            if spoken:  # the human heard it and did not answer; there is just no screen too
                log.warning("no answer by voice, and the keyboard dialog cannot open: %s", exc)
                return None
            raise AskFailed(f"cannot reach the human: no working voice, and {exc}") from exc
        if text is None or not text.strip():
            log.info("the question was dismissed")
            return None
        return _Reply(text.strip(), "keyboard", anyio.current_time())

    async def _by_voice(
        self, question: str, context: str | None, options: list[str]
    ) -> tuple[str | None, bool]:
        """(what was heard or None, whether the question was spoken at all)."""
        voice = self._voice
        if voice is None or not self._settings.voice:
            return None, False
        spoken = False
        try:
            ready_wait = min(VOICE_READY_WAIT_S, self._settings.ask_timeout / 3)
            if not await voice.wait_ready(ready_wait):
                log.info("voice is not ready, using the keyboard")
                return None, False
            await voice.say(spoken_question(question, context, options))
            spoken = True
            for attempt in range(2):
                await voice.beep()
                heard = (await voice.listen(options)).strip()
                if heard:
                    return heard, True
                if attempt == 0:
                    await voice.say(f"Sorry, I didn't catch that. {_sentence(question)}")
            if self._keyboard.available():
                await voice.say("I'll show the question on your screen.")
        except Exception:
            log.warning("voice failed, using the keyboard", exc_info=True)
        return None, spoken

    async def _confirm(self, voice: VoiceChannel, heard: str, options: list[str]) -> None:
        matched = match_option(heard, options) if options else None
        try:
            await voice.say(f"Got it: {matched}." if matched else "Got it.")
        except Exception:
            log.debug("could not confirm the answer aloud", exc_info=True)

    async def _report_progress(self, progress: Progress, started: float) -> None:
        total = self._settings.ask_timeout
        while True:
            await anyio.sleep(self._settings.progress_interval)
            elapsed = anyio.current_time() - started
            try:
                await progress(elapsed, total, f"Waiting for the human ({elapsed:.0f} s)")
            except Exception:
                log.debug("progress notification failed", exc_info=True)


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - anyio.current_time())
