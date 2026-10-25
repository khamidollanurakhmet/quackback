"""The commit-msg hook strips AI co-author trailers and leaves everything else alone.

Each case makes a real commit in a throwaway repository, so git runs the hook exactly as it
would in a clone with `core.hooksPath=.githooks`. Messages are bytes end to end, so no
platform encoding gets in the way.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parent.parent / ".githooks"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    return path


def _git(repo: Path, *args: str, locale: str | None = None) -> bytes:
    # Drop inherited GIT_* variables: run from inside a git hook or a linked worktree, a
    # leaked GIT_DIR / GIT_INDEX_FILE would point these commands at the developer's repo.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # Isolate from the developer's global and system config (signing, templates, hooks).
    global_config = repo.parent / "gitconfig"
    global_config.touch()
    env |= {
        "GIT_CONFIG_GLOBAL": str(global_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    if locale is not None:
        env["LC_ALL"] = locale
    result = subprocess.run(
        ["git", "-c", f"core.hooksPath={HOOKS_DIR.as_posix()}", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _commit(repo: Path, message: bytes, locale: str | None = None) -> bytes:
    """Commit `message` through the hook and return the message git stored."""
    message_file = repo.parent / "message.txt"
    message_file.write_bytes(message)
    _git(repo, "commit", "-q", "--allow-empty", "-F", str(message_file), locale=locale)
    raw = _git(repo, "cat-file", "commit", "HEAD")
    return raw.split(b"\n\n", 1)[1].rstrip(b"\n")


@pytest.mark.parametrize(
    "trailer",
    [
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
        "co-authored-by: claude <noreply@anthropic.com>",
        "Co-authored-by: Some Bot <bot@anthropic.com>",
        "Claude-Session: 0123456789abcdef",
        "Generated with [Claude Code](https://claude.com/claude-code)",
        "\U0001f916 Generated with [Claude Code](https://claude.com/claude-code)",
    ],
)
def test_strips_ai_trailers(repo: Path, trailer: str) -> None:
    message = f"feat: add thing\n\nBody line stays.\n\n{trailer}\n".encode()
    assert _commit(repo, message) == b"feat: add thing\n\nBody line stays."


def test_strips_all_trailers_together(repo: Path) -> None:
    message = (
        b"fix: handle it\n\nWhy it broke.\n\n"
        b"Generated with [Claude Code](https://claude.com/claude-code)\n\n"
        b"Co-Authored-By: Claude <noreply@anthropic.com>\n"
        b"Claude-Session: abc\n"
    )
    assert _commit(repo, message) == b"fix: handle it\n\nWhy it broke."


def test_keeps_human_co_author(repo: Path) -> None:
    message = b"docs: fix typo\n\nCo-Authored-By: Jane Doe <jane@example.com>"
    assert _commit(repo, message) == message


def test_leaves_plain_message_unchanged(repo: Path) -> None:
    message = "chore: bump deps\n\nNothing else to say, café.".encode()
    assert _commit(repo, message) == message


def test_keeps_non_utf8_lines_in_utf8_locale(repo: Path) -> None:
    # A Latin-1 byte once made grep treat the whole message as binary and drop the line.
    # git then stores the message as UTF-8, reading the stray byte as Latin-1.
    stored = _commit(repo, b"feat: x\n\nbody caf\xe9\nkeep me", locale="C.UTF-8")
    assert stored == "feat: x\n\nbody café\nkeep me".encode()
