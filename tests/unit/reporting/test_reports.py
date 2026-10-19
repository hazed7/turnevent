from pathlib import Path

import pytest

from adapter.ledger import PublicationLedger
from adapter.publisher import Mode, OutboxFile, Publisher
from artifact.bundle import LoadedBundle, load_bundle
from core.runtime import describe_runtime
from dataset.audit import build_data_audit
from dataset.cleaning import clean_historical
from dataset.loader import load_historical
from dataset.synthetic import SyntheticConfig, write_synthetic_csv
from inference.runner import run_inference
from labels.definition import summarize_labels
from labels.generator import LabelConfig, generate_labels
from labels.sensitivity import label_sensitivity
from reporting.data_audit import render_data_audit
from reporting.evaluation import render_evaluation
from reporting.model_card import render_model_card
from reporting.online import render_online_inference
from training.pipeline import Predictor, TrainingResult

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def bundle(trained_bundle_dir: Path) -> LoadedBundle:
    return load_bundle(trained_bundle_dir)


def test_data_audit_covers_the_sow_checklist(tmp_path: Path) -> None:
    source = tmp_path / "history.csv"
    write_synthetic_csv(SyntheticConfig(n_symbols=1, bars_per_symbol=500), source)
    raw = load_historical(source)
    clean = clean_historical(raw.frame)
    labels = generate_labels(clean, LabelConfig())
    text = render_data_audit(
        build_data_audit(raw, clean, describe_runtime(PROJECT_ROOT)),
        summarize_labels(labels),
        LabelConfig(),
        label_sensitivity(clean, LabelConfig(), {"threshold_coefficient": (0.5, 0.65)}),
    )
    for heading in ("signal_barrier", "Чистка", "Периодичность", "Разметка", "Чувствительность"):
        assert f"## {heading}" in text
    assert raw.source.sha256 in text
    assert "`signal_barrier` не читается в память" in text


def test_evaluation_compares_model_baselines_and_lags(trained_result: TrainingResult) -> None:
    by_lag = {3: trained_result.report.test_event[Predictor.MODEL]}
    text = render_evaluation("tm-x", trained_result.report, trained_result.split, by_lag)
    for row in ("majority (no-skill)", "rule: rd_value cross", "| модель |"):
        assert row in text
    assert "## Confusion matrix" in text
    assert "## Момент сигнала" in text
    assert "синтетическом" in text


def test_model_card_states_signal_semantics(bundle: LoadedBundle) -> None:
    text = render_model_card(bundle.manifest)
    assert bundle.manifest.model_version in text
    assert "windowEndTimestamp − 3 мин" in text
    assert "не торговая система" in text


def test_online_report_lists_every_window(bundle: LoadedBundle, tmp_path: Path) -> None:
    payload = (PROJECT_ROOT / "fixtures" / "online-invalid.json").read_bytes()
    publisher = Publisher(Mode.DRY_RUN, PublicationLedger(), OutboxFile(tmp_path / "out.jsonl"))
    decisions = publisher.publish(run_inference(payload, bundle))
    text = render_online_inference({"online-invalid.json": decisions}, "tm-x")
    for symbol in (f"BAD00{n}" for n in range(1, 8)):
        assert symbol in text
    assert "invalid_input → not_publishable | 7" in text
