# quackback

*Rubber duck debugging, reversed. When your coding agent gets stuck, the duck waddles over and asks you.*

quackback is an [MCP](https://modelcontextprotocol.io) server with one tool, `ask_human`. When a
coding agent has a question, it calls the tool: quackback asks you out loud, listens to your spoken
answer and hands the text back to the agent. With a [Microduck](https://github.com/pollen-robotics/microduck)
robot, the duck walks over to you first; without one, your PC does the asking.

> **Status:** voice on the PC works (speaker + microphone, with a keyboard fallback). The robot
> comes next.

## Try it

```bash
uv tool install git+https://github.com/khamidollanurakhmet/quackback
quackback download   # the voice models, about 550 MB, once
quackback try        # asks a test question out loud and prints what it heard
```

The MCP server is the plain `quackback` command on stdio. Point your MCP client at it; the
`ask_human` call blocks until you answer, for up to `QUACKBACK_ASK_TIMEOUT` seconds (300 by
default), so give the client a longer tool timeout than that.

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

[Apache-2.0](LICENSE)
