import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from adapter.ledger import PublicationLedger
from adapter.publisher import Decision, Mode, OutboxFile, Publisher
from artifact.bundle import MANIFEST_FILE, load_bundle
from artifact.manifest import ModelManifest
from cli.config import PipelineConfig
from core.errors import ArtifactError, ConfigError
from core.files import write_json, write_text_atomic
from core.runtime import describe_runtime
from dataset.audit import build_data_audit
from dataset.cleaning import CleanDataset, clean_historical
from dataset.loader import RawDataset, load_historical
from dataset.manifest import build_dataset_manifest
from dataset.synthetic import write_synthetic_csv
from inference.runner import run_inference
from inference.source import fetch_envelope, read_fixture
from labels.definition import build_label_definition, summarize_labels
from labels.generator import generate_labels
from labels.sensitivity import label_sensitivity
from reporting.data_audit import render_data_audit
from reporting.evaluation import render_evaluation
from reporting.model_card import render_model_card
from reporting.online import render_online_inference
from training.export import (
    FEATURE_SCHEMA_FILE,
    LABEL_DEFINITION_FILE,
    SPLIT_MANIFEST_FILE,
    export_bundle,
)
from training.pipeline import Predictor, train_and_evaluate

LATEST_FILE: Final = "LATEST"
SMOKE_FIXTURES: Final = ("online-ready.json", "online-warmup.json", "online-invalid.json")
COMPARED_LAG: Final = 0


def synthetic(config: PipelineConfig) -> Path:
    write_synthetic_csv(config.synthetic, config.paths.dataset)
    return config.paths.dataset


def audit(config: PipelineConfig) -> Path:
    raw, clean = _load(config)
    labels = generate_labels(clean, config.training.labels)
    write_json(
        config.paths.manifests / "dataset-manifest.json",
        build_dataset_manifest(raw, clean).model_dump(mode="json"),
    )
    write_json(
        config.paths.manifests / LABEL_DEFINITION_FILE,
        build_label_definition(labels, raw.source.sha256).model_dump(mode="json"),
    )
    report = config.paths.reports / "data-audit.md"
    write_text_atomic(
        report,
        render_data_audit(
            build_data_audit(raw, clean, describe_runtime(Path.cwd())),
            summarize_labels(labels),
            config.training.labels,
            label_sensitivity(clean, config.training.labels),
        ),
    )
    return report


def train(config: PipelineConfig) -> ModelManifest:
    raw, clean = _load(config)
    result = train_and_evaluate(clean, config.training)
    early = train_and_evaluate(
        clean, config.training.model_copy(update={"prediction_lag_bars": COMPARED_LAG})
    )
    directory = config.paths.artifacts / "staging"
    manifest = export_bundle(
        result,
        config.training,
        build_dataset_manifest(raw, clean),
        describe_runtime(Path.cwd()),
        directory,
    )
    bundle = config.paths.artifacts / manifest.model_version
    if bundle.exists():
        shutil.rmtree(directory)
    else:
        directory.rename(bundle)
    write_text_atomic(config.paths.artifacts / LATEST_FILE, manifest.model_version + "\n")
    for name in (MANIFEST_FILE, FEATURE_SCHEMA_FILE, LABEL_DEFINITION_FILE, SPLIT_MANIFEST_FILE):
        target = config.paths.manifests / ("model-manifest.json" if name == MANIFEST_FILE else name)
        write_text_atomic(target, (bundle / name).read_text(encoding="utf-8"))
    by_lag = {
        config.training.prediction_lag_bars: result.report.test_event[Predictor.MODEL],
        COMPARED_LAG: early.report.test_event[Predictor.MODEL],
    }
    write_text_atomic(
        config.paths.reports / "evaluation.md",
        render_evaluation(manifest.model_version, result.report, result.split, by_lag),
    )
    write_text_atomic(config.paths.reports / "model-card.md", render_model_card(manifest))
    return manifest


def infer(
    config: PipelineConfig,
    *,
    fixture: Path | None,
    url: str | None,
    mode: Mode,
    bundle: Path | None = None,
) -> tuple[Decision, ...]:
    payload = read_fixture(fixture) if fixture else fetch_envelope(_required(url), config.retry)
    loaded = load_bundle(bundle or latest_bundle(config))
    runs = config.paths.runs
    publisher = Publisher(
        mode, PublicationLedger(runs / "ledger.jsonl"), OutboxFile(runs / "outbox.jsonl")
    )
    decisions = publisher.publish(run_inference(payload, loaded))
    _append_jsonl(runs / "telemetry.jsonl", (d.model_dump(mode="json") for d in decisions))
    return decisions


def smoke(config: PipelineConfig, bundle: Path | None = None) -> Path:
    loaded = load_bundle(bundle or latest_bundle(config))
    publisher = Publisher(
        Mode.DRY_RUN, PublicationLedger(), OutboxFile(config.paths.runs / "outbox.jsonl")
    )
    runs = {
        name: publisher.publish(run_inference(read_fixture(config.paths.fixtures / name), loaded))
        for name in SMOKE_FIXTURES
    }
    report = config.paths.reports / "online-inference.md"
    write_text_atomic(report, render_online_inference(runs, loaded.manifest.model_version))
    return report


def run_all(config: PipelineConfig) -> ModelManifest:
    synthetic(config)
    audit(config)
    manifest = train(config)
    smoke(config)
    return manifest


def latest_bundle(config: PipelineConfig) -> Path:
    pointer = config.paths.artifacts / LATEST_FILE
    try:
        return config.paths.artifacts / pointer.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ArtifactError(f"no trained bundle yet ({pointer} is missing): run `train`") from exc


def _load(config: PipelineConfig) -> tuple[RawDataset, CleanDataset]:
    raw = load_historical(config.paths.dataset)
    return raw, clean_historical(raw.frame)


def _required(url: str | None) -> str:
    if not url:
        raise ConfigError("pass a fixture path or a DS API url")
    return url


def _append_jsonl(path: Path, records: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
