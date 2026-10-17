from contracts.market import FEATURE_COLUMNS, WINDOW_LENGTH, Column
from features.builder import FEATURE_BUILDER_VERSION, FEATURE_NAMES
from features.schema import FeatureSchema, build_feature_schema


def test_schema_lists_every_feature_once_in_build_order() -> None:
    schema = build_feature_schema()
    names = [feature.name for feature in schema.features]
    assert names == list(FEATURE_NAMES)
    assert len(set(names)) == len(names)
    assert all(feature.description for feature in schema.features)


def test_schema_pins_the_online_input_contract() -> None:
    schema = build_feature_schema()
    assert schema.builder_version == FEATURE_BUILDER_VERSION
    assert schema.window_length == WINDOW_LENGTH
    assert schema.input_columns == FEATURE_COLUMNS
    assert Column.SIGNAL_BARRIER not in schema.input_columns
    assert "nothing is fitted" in schema.normalization


def test_schema_round_trips_and_fingerprints_stably() -> None:
    schema = build_feature_schema()
    dumped = schema.model_dump(mode="json")
    assert FeatureSchema.model_validate(dumped, strict=False) == schema
    assert build_feature_schema().fingerprint() == schema.fingerprint()
