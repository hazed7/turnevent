import json
from collections.abc import Callable
from pathlib import Path

import pytest

from adapter.publisher import Mode, Outcome
from cli import commands
from cli.config import PipelineConfig, load_config
from core.errors import ArtifactError, ConfigError
from dataset.synthetic import SyntheticConfig, write_synthetic_csv


@pytest.fixture
def config(config_file: Path) -> PipelineConfig:
    return load_config(config_file)


@pytest.fixture
def trained(config: PipelineConfig) -> PipelineConfig:
    commands.demo(config)
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


def test_all_does_not_change_existing_dataset(trained: PipelineConfig) -> None:
    original = trained.paths.dataset.read_bytes()
    other_seed = trained.synthetic.model_copy(update={"seed": trained.synthetic.seed + 1})
    commands.run_all(trained.model_copy(update={"synthetic": other_seed}))
    assert trained.paths.dataset.read_bytes() == original


@pytest.mark.parametrize("command", [commands.synthetic, commands.demo])
def test_synthetic_commands_preserve_existing_dataset(
    config: PipelineConfig, command: Callable[[PipelineConfig], object]
) -> None:
    config.paths.dataset.parent.mkdir(parents=True)
    config.paths.dataset.write_text("customer bytes", encoding="utf-8")
    with pytest.raises(ConfigError, match="refusing to overwrite"):
        command(config)
    assert config.paths.dataset.read_text(encoding="utf-8") == "customer bytes"


def test_synthetic_requires_csv_path(config: PipelineConfig) -> None:
    paths = config.paths.model_copy(
        update={"dataset": config.paths.dataset.with_suffix(".parquet")}
    )
    config = config.model_copy(update={"paths": paths})
    with pytest.raises(ConfigError, match=r"new \.csv dataset path"):
        commands.synthetic(config)
    assert not config.paths.dataset.exists()


def test_synthetic_refuses_dangling_symlink(config: PipelineConfig, tmp_path: Path) -> None:
    config.paths.dataset.parent.mkdir(parents=True)
    target = tmp_path / "absent.csv"
    config.paths.dataset.symlink_to(target)
    with pytest.raises(ConfigError, match="refusing to overwrite"):
        commands.synthetic(config)
    assert config.paths.dataset.is_symlink()
    assert not target.exists()


def test_synthetic_preserves_file_created_during_generation(
    config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_writer = write_synthetic_csv

    def competing_writer(synthetic_config: SyntheticConfig, path: Path) -> None:
        original_writer(synthetic_config, path)
        config.paths.dataset.write_text("concurrent data", encoding="utf-8")

    monkeypatch.setattr(commands, "write_synthetic_csv", competing_writer)
    with pytest.raises(ConfigError, match="refusing to overwrite"):
        commands.synthetic(config)
    assert config.paths.dataset.read_text(encoding="utf-8") == "concurrent data"
    assert not list(config.paths.dataset.parent.glob(".synthetic-*"))


def test_synthetic_reports_unsupported_hard_links(
    config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(*_: object) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr("core.files.os.link", unsupported)
    with pytest.raises(ConfigError, match="cannot publish dataset"):
        commands.synthetic(config)
    assert not config.paths.dataset.exists()
    assert not list(config.paths.dataset.parent.glob(".synthetic-*"))


@pytest.mark.parametrize(
    ("filename", "dataset_name"),
    [
        ("real-data.yaml", "marketscreener-2026-10-v1"),
        ("real-data-smoke.yaml", "marketscreener-2026-10-smoke"),
    ],
)
def test_real_data_config_is_separate_and_keeps_training_defaults(
    filename: str, dataset_name: str
) -> None:
    config = load_config(Path("configs") / filename)
    assert config.training == PipelineConfig().training
    assert config.paths.dataset == Path("data/prepared") / f"{dataset_name}.parquet"
    for directory in (
        config.paths.reports,
        config.paths.manifests,
        config.paths.artifacts,
        config.paths.runs,
    ):
        assert directory.is_relative_to(Path("data/experiments") / dataset_name)
