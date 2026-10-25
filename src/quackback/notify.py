"""Best-effort desktop notifications. Never raises; a missing backend just means no toast."""

import logging
from typing import Any

import anyio

log = logging.getLogger(__name__)

_TIMEOUT_S = 5.0
_notifier: Any = None


async def send(title: str, message: str) -> bool:
    """Show a notification on the running asyncio loop. True if a real backend took it."""
    global _notifier
    try:
        with anyio.fail_after(_TIMEOUT_S):
            from desktop_notifier import DEFAULT_SOUND, DesktopNotifier, Urgency

            if _notifier is None:
                # On Windows the app name is also the toast's AppUserModelID, so keep it stable.
                _notifier = DesktopNotifier(app_name="quackback")
            # Empty capabilities: the dummy backend (macOS outside an .app bundle).
            if not await _notifier.get_capabilities():
                return False
            # Without an explicit sound, a Windows toast is silent.
            await _notifier.send(
                title=title, message=message, urgency=Urgency.Critical, sound=DEFAULT_SOUND
            )
            return True
    except Exception:  # ImportError, WinRT/registry errors, D-Bus errors, TimeoutError
        log.warning("desktop notification unavailable", exc_info=True)
        return False
