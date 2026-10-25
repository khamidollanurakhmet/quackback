# quackback

*Rubber duck debugging, reversed. When your coding agent gets stuck, the duck waddles over and asks you.*

quackback is an [MCP](https://modelcontextprotocol.io) server with one tool, `ask_human`. When a
coding agent has a question, it calls the tool: quackback asks you out loud, listens to your spoken
answer and hands the text back to the agent. With a [Microduck](https://github.com/pollen-robotics/microduck)
robot, the duck walks over to you first; without one, your PC does the asking.

> **Status:** early development — nothing to install yet.

## Development

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
```

Enable the commit-msg hook once per clone:

```bash
git config core.hooksPath .githooks
```

## Credits

quackback builds on [Microduck](https://github.com/pollen-robotics/microduck) by Pollen Robotics /
Hugging Face (Apache-2.0).

## License

[Apache-2.0](LICENSE)
