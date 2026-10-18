import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from artifact.bundle import load_bundle
from core.errors import ArtifactError
from training.pipeline import TrainingResult

type Edit = Callable[[dict[str, Any]], None]

SRC = Path(__file__).parents[3] / "src"


def _edit_manifest(directory: Path, edit: Edit) -> None:
    path = directory / "model-manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def test_loaded_bundle_predicts_like_the_trained_models(
    trained_result: TrainingResult, sample_features: pd.DataFrame, bundle_dir: Path
) -> None:
    bundle = load_bundle(bundle_dir)
    for loaded, original in (
        (bundle.event_classifier, trained_result.event_classifier),
        (bundle.state_classifier, trained_result.state_classifier),
    ):
        pd.testing.assert_frame_equal(
            loaded.probabilities(sample_features), original.probabilities(sample_features)
        )
        assert loaded.turn_threshold == original.turn_threshold


def test_bundle_loads_in_a_fresh_interpreter_without_training_code(bundle_dir: Path) -> None:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from artifact.bundle import load_bundle\n"
        f"bundle = load_bundle(Path({str(bundle_dir)!r}))\n"
        "assert not [m for m in sys.modules if m.split('.')[0] == 'training']\n"
        "print(bundle.manifest.model_version)\n"
    )
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(SRC)}
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("tm-")


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (lambda m: m.update(window_length=32), "window_length"),
        (lambda m: m.update(feature_schema_version="0.9.0"), "feature_schema_version"),
        (lambda m: m.update(feature_names=m["feature_names"][:-1]), "feature_names"),
        (lambda m: m.update(artifact_kind="something-else"), "artifact_kind"),
        (lambda m: m["runtime"]["packages"].update({"scikit-learn": "0.1"}), "scikit-learn"),
    ],
)
def test_incompatible_bundles_are_refused(bundle_dir: Path, edit: Edit, field: str) -> None:
    _edit_manifest(bundle_dir, edit)
    with pytest.raises(ArtifactError, match=field):
        load_bundle(bundle_dir)


@pytest.mark.parametrize("name", ["event-model.joblib", "label-definition.json"])
def test_tampered_files_are_refused_before_unpickling(bundle_dir: Path, name: str) -> None:
    path = bundle_dir / name
    path.write_bytes(path.read_bytes() + b" ")
    if name.endswith(".json"):
        document = json.loads(path.read_text())
        document["dataset_sha256"] = "0" * 64
        path.write_text(json.dumps(document))
    with pytest.raises(ArtifactError, match="checksum"):
        load_bundle(bundle_dir)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("model-manifest.json", "unreadable model manifest"),
        ("state-model.joblib", "missing model file"),
        ("split-manifest.json", "unreadable bundle document"),
    ],
)
def test_missing_pieces_are_reported(bundle_dir: Path, missing: str, message: str) -> None:
    (bundle_dir / missing).unlink()
    with pytest.raises(ArtifactError, match=message):
        load_bundle(bundle_dir)
