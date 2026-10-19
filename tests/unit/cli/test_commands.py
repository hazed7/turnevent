import json
from pathlib import Path

import pytest

from adapter.publisher import Mode, Outcome
from cli import commands
from cli.config import PipelineConfig, load_config
from core.errors import ArtifactError, ConfigError


@pytest.fixture
def config(config_file: Path) -> PipelineConfig:
    return load_config(config_file)


@pytest.fixture
def trained(config: PipelineConfig) -> PipelineConfig:
    commands.run_all(config)
    return config


def _lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_run_produces_the_sow_delivery_layout(trained: PipelineConfig) -> None:
    paths = trained.paths
    assert {p.name for p in paths.manifests.iterdir()} == {
        "dataset-manifest.json",
        "feature-schema.json",
        "label-definition.json",
        "model-manifest.json",
        "split-manifest.json",
    }
    assert {p.name for p in paths.reports.iterdir()} == {
        "data-audit.md",
        "evaluation.md",
        "model-card.md",
        "online-inference.md",
    }
    version = (paths.artifacts / "LATEST").read_text(encoding="utf-8").strip()
    assert (paths.artifacts / version / "event-model.joblib").exists()
    assert not (paths.artifacts / "staging").exists()


def test_retraining_the_same_data_reuses_the_same_bundle(trained: PipelineConfig) -> None:
    first = commands.latest_bundle(trained)
    assert commands.train(trained).model_version == first.name
    assert not (trained.paths.artifacts / "staging").exists()


def test_shadow_inference_publishes_once_and_logs_telemetry(trained: PipelineConfig) -> None:
    fixture = trained.paths.fixtures / "online-ready.json"
    first = commands.infer(trained, fixture=fixture, url=None, mode=Mode.SHADOW)
    again = commands.infer(trained, fixture=fixture, url=None, mode=Mode.SHADOW)

    published = [d for d in first if d.outcome is Outcome.PUBLISHED]
    assert all(d.outcome is not Outcome.PUBLISHED for d in again)
    assert len(_lines(trained.paths.runs / "outbox.jsonl")) == len(published)
    assert len(_lines(trained.paths.runs / "telemetry.jsonl")) == len(first) + len(again)


def test_inference_can_read_the_ds_api(
    trained: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = (trained.paths.fixtures / "online-warmup.json").read_bytes()
    calls: list[str] = []

    def fake_fetch(url: str, _policy: object) -> bytes:
        calls.append(url)
        return envelope

    monkeypatch.setattr(commands, "fetch_envelope", fake_fetch)
    decisions = commands.infer(trained, fixture=None, url="http://ds-api.local", mode=Mode.DRY_RUN)
    assert calls == ["http://ds-api.local"]
    assert {d.outcome for d in decisions} == {Outcome.NOT_PUBLISHABLE}


def test_inference_needs_a_trained_bundle(config: PipelineConfig) -> None:
    fixture = config.paths.fixtures / "online-ready.json"
    with pytest.raises(ArtifactError, match="run `train`"):
        commands.infer(config, fixture=fixture, url=None, mode=Mode.DRY_RUN)


def test_inference_needs_a_source(config: PipelineConfig) -> None:
    with pytest.raises(ConfigError, match="fixture path or a DS API url"):
        commands.infer(config, fixture=None, url=None, mode=Mode.DRY_RUN)


def test_repository_config_matches_the_defaults() -> None:
    assert load_config(Path("configs/pipeline.yaml")) == PipelineConfig()
