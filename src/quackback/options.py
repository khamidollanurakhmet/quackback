"""Match a spoken or typed answer to the closest of the offered options.

Speech is messy ("uh, the second one", "let's go with postgres", "Reddis"), so matching tries, in
order: the whole answer is an option; the answer names exactly one option; the answer points at
an option by number; a close fuzzy match. A named or fuzzy match must also make up most of what
was said, so "no idea, you decide" does not become the option "no". Anything unclear returns
None, and the caller hands the raw transcript to the agent instead of guessing.
"""

import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import pairwise

_NUMBERS = {
    "one": 1, "first": 1, "1": 1, "1st": 1,
    "two": 2, "second": 2, "2": 2, "2nd": 2,
    "three": 3, "third": 3, "3": 3, "3rd": 3,
    "four": 4, "fourth": 4, "4": 4, "4th": 4,
    "five": 5, "fifth": 5, "5": 5, "5th": 5,
    "six": 6, "sixth": 6, "6": 6, "6th": 6,
    "seven": 7, "seventh": 7, "7": 7, "7th": 7,
    "eight": 8, "eighth": 8, "8": 8, "8th": 8,
    "nine": 9, "ninth": 9, "9": 9, "9th": 9,
    "ten": 10, "tenth": 10, "10": 10, "10th": 10,
}  # fmt: skip

# Words that carry no choice: "um, I'd say the second one please" -> "second".
_FILLER = {
    "um", "uh", "erm", "hmm", "ok", "okay", "so", "well", "yeah", "yes", "i", "id", "d", "say",
    "think", "guess", "would", "go", "with", "lets", "let", "s", "us", "pick", "choose", "take",
    "the", "a", "an", "option", "number", "choice", "one", "please", "thanks", "thank", "you",
}  # fmt: skip

# Words that turn an answer against what it names: "not Redis", "don't delete the branch".
# "don't" normalizes to "don t", so the "t" of any n't contraction counts.
_NEGATIONS = {
    "not", "no", "never", "t", "dont", "cant", "wont", "nope", "nah", "none", "neither", "nor",
    "without", "except", "skip", "avoid", "instead",
}  # fmt: skip
# Stalling, not choosing: "one second", "hold on", "wait a moment".
_STALLING = {"wait", "hold", "hang", "moment", "sec", "minute"}

FUZZY_THRESHOLD = 0.8
# Shorter options must match exactly, by name or by number: "now" is not a typo of "no".
FUZZY_MIN_LENGTH = 4
# A named or fuzzy option must be at least this share of the meaningful words said.
MIN_COVERAGE = 0.5


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_only.lower()).split())


def match_option(answer: str, options: Sequence[str]) -> str | None:
    """Return the option the answer refers to, or None if there is no clear match."""
    said = normalize(answer)
    choices = [normalize(option) for option in options]
    if not said or not options:
        return None
    for strategy in (_exact, _named, _numbered, _fuzzy):
        index = strategy(said, choices)
        if index is None:
            continue
        # "Don't delete the branch" names an option only to reject it.
        if strategy is not _exact and _negated(said, choices[index]):
            return None
        return options[index]
    return None


def _negated(said: str, choice: str) -> bool:
    choice_words = set(choice.split())
    return any(word in _NEGATIONS and word not in choice_words for word in said.split())


def _exact(said: str, choices: list[str]) -> int | None:
    hits = [i for i, choice in enumerate(choices) if choice and choice == said]
    return hits[0] if len(hits) == 1 else None


def _named(said: str, choices: list[str]) -> int | None:
    """The single option whose words appear in the answer; the longest one wins on overlap."""
    hits = [
        i for i, choice in enumerate(choices)
        if choice and re.search(rf"\b{re.escape(choice)}\b", said) and _covers(said, choice)
    ]  # fmt: skip
    if not hits:
        return None
    longest = max(hits, key=lambda i: len(choices[i]))
    # "postgres 16" contains "postgres": fine. "postgres or sqlite" names two options: ambiguous.
    if all(choices[i] in choices[longest] for i in hits):
        return longest
    return None


def _numbered(said: str, choices: list[str]) -> int | None:
    """An answer like "option two", "the second one", "number 2" or "the last one"."""
    tokens = said.split()
    if _STALLING & set(tokens) or any(
        first == "one" and second in _NUMBERS for first, second in pairwise(tokens)
    ):
        return None  # "one second", "hold on a moment": not a choice
    words = [word for word in tokens if word not in _FILLER or word == "one"]
    # "one" is both a number and filler ("the second one"); it counts only when it is alone.
    if len(words) > 1:
        words = [word for word in words if word != "one"]
    if len(words) != 1:
        return None
    if words[0] == "last":
        return len(choices) - 1
    number = _NUMBERS.get(words[0])
    if number is None or number > len(choices):
        return None
    return number - 1


def _fuzzy(said: str, choices: list[str]) -> int | None:
    scores = [
        _similarity(said, choice)
        if len(choice) >= FUZZY_MIN_LENGTH and _covers(said, choice)
        else 0.0
        for choice in choices
    ]
    best = max(range(len(choices)), key=lambda i: scores[i])
    runner_up = max((score for i, score in enumerate(scores) if i != best), default=0.0)
    if scores[best] >= FUZZY_THRESHOLD and scores[best] - runner_up >= 0.1:
        return best
    return None


def _covers(said: str, choice: str) -> bool:
    choice_words = choice.split()
    meaningful = [word for word in said.split() if word not in _FILLER or word in choice_words]
    return bool(meaningful) and len(choice_words) / len(meaningful) >= MIN_COVERAGE


def _similarity(said: str, choice: str) -> float:
    """Best similarity between the option and any same-length run of words in the answer."""
    if not choice:
        return 0.0
    words, size = said.split(), len(choice.split())
    if len(words) <= size:
        return SequenceMatcher(None, said, choice).ratio()
    windows = (" ".join(words[i : i + size]) for i in range(len(words) - size + 1))
    return max(SequenceMatcher(None, window, choice).ratio() for window in windows)
