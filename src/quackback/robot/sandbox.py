"""The Microduck simulator website as a duck: https://huggingface.co/spaces/pollen-robotics/microduck-simulator

That page runs MuJoCo and the real trained policies in the browser, with no server and no
JSON-RPC. It does publish debug hooks for automated testing (`window.rl`): the MuJoCo state and
the input Controller, which takes extra input sources the way it takes a gamepad. quackback opens
the page in a browser it drives (Playwright) and adds itself as one more source:

- its twist drives the policy only while no keyboard, pad or touch input is active, so a human at
  the keys always wins, as on the real duck;
- like robotd's deadman, a twist not renewed within 500 ms is dropped, so the duck stops if
  quackback stops;
- the pose is the duck's MuJoCo free joint, the simulator's ground truth;
- the page's multiplayer ghosts are switched off (its relays are refused), so nobody else
  visiting the website sees quackback's duck wander around.

The page's walking policy stands still below 0.25 m/s and turns only while walking (measured),
which is what `GAIT` says.
"""

import logging
import math
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import anyio

from quackback.robot.base import Pose, RobotError
from quackback.robot.microduck import Gait

log = logging.getLogger(__name__)

GAIT = Gait(max_speed=0.25, max_turn=1.0, min_speed=0.25, turns_in_place=False)
# The arena's walls (constants.js ARENA_HALF = 1.5 m), in the page's ground frame.
ARENA = (-1.5, 1.5, -1.5, 1.5)
BOOT_TIMEOUT_MS = 180_000
_MISSING = "quackback-hook-missing"
_CALL = f"""([name, arg]) =>
  window.__quackback ? window.__quackback[name](arg) : "{_MISSING}" """
VIEWPORT = {"width": 1280, "height": 720}

# Installed once the simulator is up. Arguments come in through evaluate(), never by pasting
# text into the script, so a question can never become code.
_INSTALL = """() => {
  if (window.__quackback) return;
  const rl = window.rl;
  const source = {
    id: "quackback", connected: true, command: new Float32Array(3),
    axes: { jaw: 0, orbitX: 0, orbitY: 0, ride: 0 }, pressed: {},
    renewedAt: -Infinity,
    // The deadman lives here, not in poll(): the 50 Hz control loop asks every source's
    // isActive() on every step, while poll() only runs when a frame is rendered.
    isActive() {
      const live = performance.now() - this.renewedAt < 500;
      if (!live) this.command.fill(0);
      return live;
    },
    init() {}, dispose() {}, poll() {},
    onAction: () => {},
  };
  rl.controller.addSource(source);
  const caption = document.createElement("div");
  caption.id = "quackback-caption";
  Object.assign(caption.style, {
    position: "fixed", top: "24px", left: "50%", transform: "translateX(-50%)",
    maxWidth: "70vw", padding: "14px 22px", borderRadius: "14px", zIndex: 99999,
    background: "rgba(255, 248, 225, 0.95)", color: "#222", display: "none",
    font: "600 20px/1.35 system-ui, sans-serif", boxShadow: "0 6px 24px rgba(0,0,0,.25)",
    pointerEvents: "none", textAlign: "center",
  });
  document.body.appendChild(caption);
  window.__quackback = {
    move([vx, vy, vyaw]) {
      source.command[0] = vx; source.command[1] = vy; source.command[2] = vyaw;
      source.renewedAt = performance.now();
    },
    stop() { source.command.fill(0); source.renewedAt = -Infinity; },
    quack() { source.onAction("quack"); },
    pose() { return Array.from(rl.data.qpos.slice(0, 7)); },
    caption(text) {
      caption.textContent = text || "";
      caption.style.display = text ? "block" : "none";
    },
  };
}"""

_BOOTED = """() => {
  const state = window.__store?.getState();
  if (state?.bootFailed) throw new Error("the simulator failed to boot");
  return !!(window.rl && state?.bootDone && !window.rl.inputLocked);
}"""


async def _refuse_websocket(route: Any) -> None:
    await route.close()


def with_boot_flag(url: str) -> str:
    """The page with ?boot=1: straight into the simulator, no title screen to click through."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.setdefault("boot", "1")
    return urlunsplit(parts._replace(query=urlencode(query)))


def pose_from_qpos(qpos: list[float]) -> Pose:
    """MuJoCo free joint [x, y, z, qw, qx, qy, qz] to a ground pose."""
    x, y, _z, w, qx, qy, qz = qpos
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return Pose(x, y, yaw)


def browser_channels(requested: str, platform: str = sys.platform) -> list[str | None]:
    """Browsers to try in order; None is Playwright's own Chromium."""
    if requested:
        return [requested]
    preferred = "msedge" if platform == "win32" else "chrome"
    return [preferred, None]


class SandboxLink:
    gait = GAIT
    bounds = ARENA

    def __init__(
        self,
        url: str,
        browser: str = "",
        headless: bool = False,
        record_dir: Path | None = None,
    ) -> None:
        self._url = with_boot_flag(url)
        self._browser_name = browser
        self._headless = headless
        self._record_dir = record_dir
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    async def open(self) -> None:
        from playwright.async_api import async_playwright

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._launch()
            options: dict[str, Any] = {"viewport": VIEWPORT}
            if self._record_dir is not None:
                self._record_dir.mkdir(parents=True, exist_ok=True)
                options |= {
                    "record_video_dir": str(self._record_dir),
                    "record_video_size": VIEWPORT,
                }
            self._context = await self._browser.new_context(**options)
            # The page's only WebSockets are its multiplayer relays (public Nostr servers that
            # show every visitor's duck to the others): quackback's duck stays private.
            await self._context.route_web_socket(re.compile(".*"), _refuse_websocket)
            self._page = await self._context.new_page()
            log.info("opening the simulator at %s", self._url)
            await self._page.goto(self._url, wait_until="domcontentloaded", timeout=BOOT_TIMEOUT_MS)
            await self._install()
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self.close()
            raise

    async def _install(self) -> None:
        """Wait for the simulator to be up and add quackback to it (again, after a reload)."""
        await self._page.wait_for_function(_BOOTED, polling=500, timeout=BOOT_TIMEOUT_MS)
        await self._page.evaluate(_INSTALL)

    async def _launch(self) -> Any:
        from playwright.async_api import Error

        failures = []
        for channel in browser_channels(self._browser_name):
            try:
                options: dict[str, Any] = {"headless": self._headless}
                if channel:
                    options["channel"] = channel
                return await self._playwright.chromium.launch(**options)
            except Error as exc:
                failures.append(f"{channel or 'chromium'}: {exc.message.splitlines()[0]}")
        raise RobotError(
            "no browser for the simulator (run `quackback setup-sandbox`): " + "; ".join(failures)
        )

    async def move(self, vx: float, vy: float, vyaw: float) -> None:
        await self._call("move", [vx, vy, vyaw])

    async def stop(self) -> None:
        if self._page is not None:
            await self._call("stop")

    async def pose(self) -> Pose:
        return pose_from_qpos(await self._call("pose"))

    async def quack(self) -> None:
        await self._call("quack")

    async def show_text(self, text: str | None) -> None:
        await self._call("caption", text)

    async def close(self) -> None:
        for name in ("_context", "_browser"):
            thing = getattr(self, name)
            if thing is not None:
                try:
                    await thing.close()  # closing the context also saves the video
                except Exception:
                    log.debug("could not close the browser %s", name, exc_info=True)
        if self._playwright is not None:
            await self._playwright.stop()
        self._page = self._context = self._browser = self._playwright = None

    async def _call(self, name: str, arg: Any = None) -> Any:
        if self._page is None:
            raise RobotError("the simulator is not open")
        for attempt in range(2):
            result = await self._page.evaluate(_CALL, [name, arg])
            if result != _MISSING:
                return result
            if attempt == 0:
                # Someone reloaded the page: the duck starts over, at its spawn point.
                log.info("the simulator page was reloaded, reconnecting")
                await self._install()
        raise RobotError("lost the simulator page")
