import pytest
from pydantic import ValidationError

from contracts.market import Column
from core.errors import TrainingError
from training.samples import SampleColumn, Samples
from training.split import Part, SplitConfig, time_split


def test_parts_are_sequential_and_their_dependency_spans_never_overlap(samples: Samples) -> None:
    parts, _ = time_split(samples, SplitConfig())
    train, validation, test = parts[Part.TRAIN], parts[Part.VALIDATION], parts[Part.TEST]
    assert (
        train.keys[SampleColumn.SPAN_END_MS].max()
        < validation.keys[SampleColumn.SPAN_START_MS].min()
    )
    assert (
        validation.keys[SampleColumn.SPAN_END_MS].max()
        < test.keys[SampleColumn.SPAN_START_MS].min()
    )


def test_no_sample_lands_in_two_parts(samples: Samples) -> None:
    parts, _ = time_split(samples, SplitConfig())
    keys = [
        part.keys[[Column.SYMBOL, Column.TIMESTAMP]].to_numpy().tolist() for part in parts.values()
    ]
    flat = [tuple(key) for part_keys in keys for key in part_keys]
    assert len(flat) == len(set(flat))


def test_embargo_pushes_later_parts_back(samples: Samples) -> None:
    config = SplitConfig(embargo_bars=30)
    parts, manifest = time_split(samples, config)
    first_boundary = manifest.boundaries_ms[0]
    validation_start = parts[Part.VALIDATION].keys[SampleColumn.SPAN_START_MS].min()
    assert validation_start >= first_boundary + 30 * 60_000
    assert manifest.embargo_ms == 30 * 60_000


def test_manifest_accounts_for_every_dropped_sample(samples: Samples) -> None:
    _, manifest = time_split(samples, SplitConfig())
    assert [period.part for period in manifest.periods] == list(Part)
    kept = sum(period.samples for period in manifest.periods)
    dropped = sum(period.dropped_at_edges for period in manifest.periods)
    assert kept + dropped == len(samples)
    assert all(period.dropped_at_edges > 0 for period in manifest.periods)


def test_split_is_deterministic(samples: Samples) -> None:
    assert time_split(samples, SplitConfig())[1] == time_split(samples, SplitConfig())[1]


def test_too_little_data_for_a_part_raises(samples: Samples) -> None:
    with pytest.raises(TrainingError, match="split is empty"):
        time_split(samples.take(samples.keys.index.to_numpy() < 200), SplitConfig())


def test_fractions_must_leave_training_data() -> None:
    with pytest.raises(ValidationError, match="leave data to train"):
        SplitConfig(validation_fraction=0.5, test_fraction=0.5)
