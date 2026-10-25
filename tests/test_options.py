import pytest

from quackback.options import match_option, normalize

DATABASES = ["PostgreSQL", "SQLite", "Redis"]


def test_normalize_strips_case_accents_and_punctuation() -> None:
    assert normalize("  Café, NOW!  ") == "cafe now"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("PostgreSQL", "PostgreSQL"),
        ("sqlite.", "SQLite"),
        ("Let's go with Redis, please.", "Redis"),
        ("Um, I'd say SQLite.", "SQLite"),
        ("The second one.", "SQLite"),
        ("Option three", "Redis"),
        ("number 1", "PostgreSQL"),
        ("the last one", "Redis"),
        ("Reddis", "Redis"),
        ("postgres sequel", "PostgreSQL"),
    ],
)
def test_matches(answer: str, expected: str) -> None:
    assert match_option(answer, DATABASES) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "Redis or SQLite, I can't decide",  # names two options
        "option five",  # out of range
        "none of them, write it to a file",
        "something completely different",
    ],
)
def test_no_clear_match(answer: str) -> None:
    assert match_option(answer, DATABASES) is None


@pytest.mark.parametrize(
    ("answer", "options"),
    [
        ("Don't delete the branch.", ["delete the branch", "keep the branch"]),
        ("Do not force push.", ["force push", "open a PR"]),
        ("Not now.", ["now", "later"]),
        ("not Redis", DATABASES),
        ("no Redis", DATABASES),
        ("never Redis", DATABASES),
        ("Never deploy on Friday", ["deploy on Friday", "deploy on Monday"]),
        ("not the second one", DATABASES),
        ("anything except SQLite", DATABASES),
        ("Redis, I don't think so", DATABASES),
    ],
)
def test_a_rejected_option_is_not_chosen(answer: str, options: list[str]) -> None:
    assert match_option(answer, options) is None


def test_negation_words_inside_an_option_are_fine() -> None:
    options = ["skip the tests", "run the tests"]
    assert match_option("skip the tests", options) == "skip the tests"
    assert match_option("let's skip the tests", options) == "skip the tests"
    assert match_option("No.", ["yes", "no"]) == "no"


@pytest.mark.parametrize("answer", ["One second.", "Hold on, the second one", "wait a moment"])
def test_stalling_is_not_a_choice(answer: str) -> None:
    assert match_option(answer, ["yes", "no", "maybe"]) is None


def test_short_options_are_not_fuzzy() -> None:
    assert match_option("Now.", ["yes", "no"]) is None
    assert match_option("nor", ["yes", "no"]) is None


def test_short_option_inside_a_longer_answer_is_not_a_match() -> None:
    assert match_option("no idea, you decide", ["yes", "no"]) is None
    assert match_option("yes but only if the tests pass", ["yes", "no"]) is None


def test_short_option_with_filler_is_a_match() -> None:
    assert match_option("yes please", ["yes", "no"]) == "yes"
    assert match_option("No.", ["yes", "no"]) == "no"


def test_longest_overlapping_option_wins() -> None:
    options = ["Python", "Python 3.13"]
    assert match_option("python 3.13", options) == "Python 3.13"
    assert match_option("python", options) == "Python"


def test_number_words_inside_options_still_match_by_name() -> None:
    options = ["one worker", "two workers"]
    assert match_option("two workers", options) == "two workers"
    assert match_option("two", options) == "two workers"


def test_no_options() -> None:
    assert match_option("anything", []) is None
