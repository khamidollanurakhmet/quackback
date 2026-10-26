# quackback

*Rubber duck debugging, reversed. When your coding agent gets stuck, the duck waddles over and asks you.*

quackback is an [MCP](https://modelcontextprotocol.io) server built around one tool, `ask_human`.
When a coding agent has a question, it calls the tool: quackback asks you out loud, listens to your spoken
answer and hands the text back to the agent. With a [Microduck](https://github.com/pollen-robotics/microduck)
robot, the duck walks over to you first; without one, your PC does the asking.

> **Status:** voice on the PC works (speaker + microphone, with a keyboard fallback), and a
> simulated Microduck finds a person with its camera and walks over to ask.

## Try it

```bash
uv tool install git+https://github.com/khamidollanurakhmet/quackback
quackback download   # the voice models, about 550 MB, once
quackback try        # asks a test question out loud and prints what it heard
```

The MCP server is the plain `quackback` command on stdio. Point your MCP client at it; the
`ask_human` call blocks until you answer, for up to `QUACKBACK_ASK_TIMEOUT` seconds (300 by
default), so give the client a longer tool timeout than that.

## A duck in the simulator

No Microduck on your desk? The [Microduck simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
runs the real trained walking policy in your browser, and quackback can drive it:

```bash
uv tool install "quackback[sandbox] @ git+https://github.com/khamidollanurakhmet/quackback"
quackback setup-sandbox    # only without Edge or Chrome, or to record video
QUACKBACK_ROBOT=sandbox quackback try
```

quackback opens the simulator in Edge or Chrome and puts a stand-in person in the arena (you
cannot stand in a browser tab). For every question the duck:

1. shows the question above the scene, and looks up and around with its head camera until it
   sees the person. The pictures are analysed on your PC, with MediaPipe's person detector;
2. walks toward them, and stops well before its depth sensor has them 40 cm away: about 65 cm in
   the simulator, whose duck turns only in arcs and needs that room to turn around and leave;
3. turns its head to face them and flaps its beak. The simulator's own sound is switched off
   upstream, so your PC speaks the question;
4. walks back to where it started once you have answered or the question timed out. Before it
   turns round it looks about with its depth sensor, and it turns only the way that keeps it
   clear of you (or stays put, if neither way does).

Whenever it walks, the duck stops for anything its depth sensor sees closer than 50 cm in its
way.

If nobody comes into view within `QUACKBACK_PERSON_SEARCH_TIMEOUT` seconds, the question is
asked on your PC without the duck. Without the `vision` extra (it comes with `sandbox`), or with
`QUACKBACK_FIND_PERSON=false`, the duck walks blind to `QUACKBACK_ROBOT_TARGET` and stops 40 cm
short.

- Keep the simulator window open, not minimized: a minimized browser stops drawing, and the duck
  sees nothing.
- MediaPipe has no build for Intel Macs, so there the duck always walks blind.
- Your keyboard or gamepad always overrides quackback in the simulator, as on the real robot.
- The duck stops by itself within half a second if quackback stops sending it commands.
- The agent also gets a second tool, `stop_robot`: stop now, and stay put for the rest of the
  question.
- quackback switches the page's multiplayer off, so other visitors never see your duck.

## Settings

Environment variables, or the same names in lower case in `config.toml` (in `%APPDATA%\quackback\`
on Windows, `~/Library/Application Support/quackback/` on macOS, `~/.config/quackback/` on Linux,
or wherever `QUACKBACK_CONFIG` points):

| Variable | Default | |
|---|---|---|
| `QUACKBACK_ASK_TIMEOUT` | `300` | seconds to wait for an answer |
| `QUACKBACK_VOICE` | `true` | `false`: skip speech, use the keyboard dialog |
| `QUACKBACK_WHISPER_MODEL` | `small.en` | any faster-whisper model, e.g. `base.en`, `large-v3-turbo` |
| `QUACKBACK_LANGUAGE` | `en` | Whisper language; empty to detect |
| `QUACKBACK_PIPER_VOICE` | `en_US-lessac-medium` | a [Piper voice](https://huggingface.co/rhasspy/piper-voices) name or a path to an `.onnx` voice |
| `QUACKBACK_INPUT_DEVICE` / `QUACKBACK_OUTPUT_DEVICE` | system default | a device name or index |
| `QUACKBACK_LOG_FILE` | none | also log here (logs always go to stderr) |
| `QUACKBACK_ROBOT` | `none` | `sandbox`: the Microduck simulator walks over to ask |
| `QUACKBACK_ROBOT_TARGET` | `1.5,0.0` | where you are: metres forward, left of the duck's start. A duck that cannot look for you walks there and stops 0.4 m short |
| `QUACKBACK_FIND_PERSON` | `true` | let a duck with a camera look for you (needs the `vision` extra) |
| `QUACKBACK_PERSON_SEARCH_TIMEOUT` | `30` | seconds to look before asking on the PC instead |
| `QUACKBACK_MAX_SPEED` / `QUACKBACK_MAX_TURN` | the robot's | speed caps, m/s and rad/s (the simulator only walks at 0.25 m/s) |
| `QUACKBACK_SANDBOX_BROWSER` | Edge on Windows, else Chrome | a Playwright channel, e.g. `chrome`, `msedge` |
| `QUACKBACK_SANDBOX_HEADLESS` | `false` | run the simulator without a window |
| `QUACKBACK_SANDBOX_RECORD_DIR` | none | record the simulator as video (run `quackback setup-sandbox` once) |
| `QUACKBACK_SANDBOX_PERSON` | at `ROBOT_TARGET` | where the stand-in person stands, `forward,left` like `ROBOT_TARGET`; `none` for nobody |
| `QUACKBACK_SANDBOX_PERSON_IMAGE` | the bundled one | a picture of the stand-in person, ideally a PNG with a transparent background |

Audio is never written to disk, and the duck's camera pictures are never saved either.

**MediaPipe and Google.** Google's MediaPipe, which quackback uses to find people in the duck's
pictures, sends Google usage metrics (not pictures) when a detector is shut down, and has no
setting to turn that off. quackback never shuts its detector down: it lives as long as the
quackback process and is dropped at exit without that call. Set `QUACKBACK_FIND_PERSON=false`,
or install without the `vision` extra, to not run MediaPipe at all. quackback downloads the
detector's model (EfficientDet-Lite0, 4.6 MB) from Google's MediaPipe model storage once, with
`quackback download` or on first use, and checks it against a pinned hash.

**Voice answers are information, not permission.** Anyone in the room, or a video playing nearby,
can answer. quackback tells the agent so, and the agent must not treat a spoken answer as approval
for anything risky.

## Development

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
QUACKBACK_TEST_MODELS=1 uv run --extra vision pytest -m models   # the real voice and detector models
QUACKBACK_TEST_SANDBOX=1 uv run --extra sandbox pytest -m sandbox   # against the simulator website
uv run --extra sandbox python scripts/record_demo.py demo/   # records the demo video
```

Enable the commit-msg hook once per clone:

```bash
git config core.hooksPath .githooks
```

## Credits

quackback builds on [Microduck](https://github.com/pollen-robotics/microduck) by Pollen Robotics /
Hugging Face (Apache-2.0). Speech uses [Piper](https://github.com/OHF-Voice/piper1-gpl) and
[faster-whisper](https://github.com/SYSTRAN/faster-whisper); the duck sees with
[MediaPipe](https://ai.google.dev/edge/mediapipe) (Apache-2.0). The simulator's stand-in person
is a cutout of a CC0 photo by Mohammad Faruque on StockSnap (details in
`src/quackback/robot/assets/README.md`).

## License

quackback is [Apache-2.0](LICENSE). Note that one of its dependencies, piper-tts, is GPL-3.0
(it bundles espeak-ng).
