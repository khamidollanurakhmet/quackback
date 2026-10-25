"""Keyboard fallback: a desktop notification plus the question in a small window.

The window is `quackback.dialog` in a child process. Cancelling `ask` (the ask timeout) kills it;
if the server itself is killed outright, the dialog notices and closes on its own.
"""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence

import anyio

from quackback import notify
from quackback.dialog import DISMISSED

log = logging.getLogger(__name__)

# -P: the server's working directory is the agent's project; nothing in it may shadow the
# modules the dialog imports.
DIALOG_COMMAND = (sys.executable, "-P", "-m", "quackback.dialog")
# The dialog outlives the ask timeout by this much at most, should nobody be left to close it.
_ORPHAN_GRACE_S = 2.0


class KeyboardUnavailable(RuntimeError):
    """The dialog could not be shown (no display, no Tk, crashed)."""


class KeyboardPrompt:
    def __init__(
        self,
        command: Sequence[str] = DIALOG_COMMAND,
        notifications: bool = True,
        check_display: bool = True,
    ):
        self._command = list(command)
        self._notifications = notifications
        self._check_display = check_display

    def available(self) -> bool:
        """False when the window certainly cannot open (Tk is X11-only on Linux)."""
        no_display = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
        return not (self._check_display and no_display)

    async def ask(
        self,
        question: str,
        context: str | None,
        options: Sequence[str],
        close_after: float | None = None,
    ) -> str | None:
        """Return the typed or clicked answer, or None if the window was closed unanswered."""
        if not self.available():
            raise KeyboardUnavailable("no display ($DISPLAY is not set)")
        request = {
            "question": question,
            "context": context,
            "options": list(options),
            "close_after": None if close_after is None else close_after + _ORPHAN_GRACE_S,
            "parent_pid": os.getpid(),
        }
        result: subprocess.CompletedProcess[bytes] | None = None
        error: OSError | None = None
        # Errors are raised after the task group, not inside it (they would arrive as a group).
        async with anyio.create_task_group() as tg:
            if self._notifications:
                tg.start_soon(notify.send, "Your coding agent has a question", question)
            try:
                result = await anyio.run_process(
                    self._command,
                    input=json.dumps(request).encode(),
                    check=False,
                    # No console window flashing up when the server itself has none.
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                error = exc
            finally:
                tg.cancel_scope.cancel()  # a slow notification must not hold the answer back
        if error is not None or result is None:
            raise KeyboardUnavailable(f"cannot start the dialog: {error}") from error
        if result.returncode == DISMISSED:
            return None
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise KeyboardUnavailable(f"dialog exited with {result.returncode}: {stderr[-500:]}")
        try:
            answer = json.loads(result.stdout)["answer"]
        except (ValueError, KeyError, TypeError) as exc:
            raise KeyboardUnavailable(f"dialog returned {result.stdout[:200]!r}") from exc
        return str(answer)
