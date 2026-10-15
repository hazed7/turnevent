from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from core.frozen import FrozenModel


class Sample(FrozenModel):
    window: int
    tags: tuple[str, ...] = ()
    counts: Mapping[str, int] = {}


def test_fields_cannot_be_reassigned() -> None:
    sample = Sample(window=64)
    with pytest.raises(ValidationError, match="frozen"):
        sample.window = 32  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Sample.model_validate({"window": 64, "signal_barrier": 1})


def test_strict_mode_rejects_coercion() -> None:
    with pytest.raises(ValidationError):
        Sample.model_validate({"window": "64"})


@pytest.mark.parametrize(
    "annotation",
    [
        dict[str, int],
        list[int],
        set[int],
        tuple[list[int], ...],
        int | list[int],
        Mapping[str, list[int]],
    ],
)
def test_mutable_containers_are_rejected_at_class_definition(annotation: object) -> None:
    with pytest.raises(TypeError, match="mutable container"):
        type("Bad", (FrozenModel,), {"__annotations__": {"field": annotation}})


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    assert Sample(window=64).fingerprint() == Sample(window=64).fingerprint()
    assert Sample(window=64).fingerprint() != Sample(window=63).fingerprint()
