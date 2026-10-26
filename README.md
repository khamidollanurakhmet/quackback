# quackback

*Rubber duck debugging, reversed. When your coding agent gets stuck, the duck waddles over and asks you.*

quackback is an [MCP](https://modelcontextprotocol.io) server built around one tool, `ask_human`.
When a coding agent has a question, it calls the tool: quackback asks you out loud, listens to your spoken
answer and hands the text back to the agent. With a [Microduck](https://github.com/pollen-robotics/microduck)
robot, the duck walks over to you first; without one, your PC does the asking.

> **Status:** voice on the PC works (speaker + microphone, with a keyboard fallback), and a
> simulated Microduck walks over to ask. Finding you with the duck's camera comes next.

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

quackback opens the simulator in Edge or Chrome. For every question the duck walks toward you
(`QUACKBACK_ROBOT_TARGET`) and stops 40 cm short, flaps its beak (the simulator's own sound is
switched off upstream, so the question itself is the sound: your PC speaks it), shows the
question above the scene, and walks back to where it started once you have answered or the
question timed out.

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
| `QUACKBACK_ROBOT_TARGET` | `1.5,0.0` | where you are: metres forward, left of the duck's start (it stops 0.4 m short) |
| `QUACKBACK_MAX_SPEED` / `QUACKBACK_MAX_TURN` | the robot's | speed caps, m/s and rad/s (the simulator only walks at 0.25 m/s) |
| `QUACKBACK_SANDBOX_BROWSER` | Edge on Windows, else Chrome | a Playwright channel, e.g. `chrome`, `msedge` |
| `QUACKBACK_SANDBOX_HEADLESS` | `false` | run the simulator without a window |
| `QUACKBACK_SANDBOX_RECORD_DIR` | none | record the simulator as video (run `quackback setup-sandbox` once) |

Audio is never written to disk.

**Voice answers are information, not permission.** Anyone in the room, or a video playing nearby,
can answer. quackback tells the agent so, and the agent must not treat a spoken answer as approval
for anything risky.

## Development

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
QUACKBACK_TEST_MODELS=1 uv run pytest -m models   # with the real Piper and Whisper models
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
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## License

quackback is [Apache-2.0](LICENSE). Note that one of its dependencies, piper-tts, is GPL-3.0
(it bundles espeak-ng).
