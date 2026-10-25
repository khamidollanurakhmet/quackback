"""A quackback server whose keyboard writes junk to stdout, for test_stdio.

Run as a script by the test, over real stdio. Python-level and file-descriptor-level writes to
stdout both happen mid-call, the way a misbehaving library would do it.
"""

import os
import sys
from collections.abc import Sequence

from quackback.ask import Asker
from quackback.config import Settings
from quackback.server import create_server, run_stdio


class NoisyKeyboard:
    def available(self) -> bool:
        return True

    async def ask(
        self,
        question: str,
        context: str | None,
        options: Sequence[str],
        close_after: float | None = None,
    ) -> str:
        sys.stdout.write("stray print from a library\n")
        sys.stdout.flush()
        os.write(1, b"stray native write\n")
        return f"typed: {question}"


if __name__ == "__main__":
    settings = Settings(voice=False)
    run_stdio(create_server(settings, Asker(settings, voice=None, keyboard=NoisyKeyboard())))
