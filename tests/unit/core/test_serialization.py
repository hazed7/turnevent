import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.serialization import (
    fingerprint,
    sha256_bytes,
    sha256_file,
    to_canonical_json,
    to_pretty_json,
)

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.text()
    | st.floats(allow_nan=False, allow_infinity=False),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
)


@given(st.dictionaries(st.text(), json_values))
def test_fingerprint_ignores_key_order(value: dict[str, object]) -> None:
    reversed_order = dict(reversed(value.items()))
    assert fingerprint(value) == fingerprint(reversed_order)


def test_fingerprint_is_content_sensitive() -> None:
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="Out of range"):
        to_canonical_json({"x": bad})
    with pytest.raises(ValueError, match="Out of range"):
        to_pretty_json({"x": bad})


def test_canonical_json_is_compact_and_sorted() -> None:
    assert to_canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode()


def test_pretty_json_is_sorted_and_newline_terminated() -> None:
    assert to_pretty_json({"b": 1, "a": 2}) == '{\n  "a": 2,\n  "b": 1\n}\n'


def test_file_hash_matches_bytes_hash(tmp_path: Path) -> None:
    payload = b"timestamp,symbol\n1,A\n"
    target = tmp_path / "data.csv"
    target.write_bytes(payload)
    assert sha256_file(target) == sha256_bytes(payload)
