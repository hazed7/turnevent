import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final

import joblib
from pydantic import ValidationError

from artifact.classifier import Estimator, TurnClassifier
from artifact.manifest import ARTIFACT_KIND, BUNDLE_FORMAT_VERSION, ModelFile, ModelManifest
from contracts.market import WINDOW_LENGTH
from core.errors import ArtifactError
from core.files import write_bytes_atomic, write_json
from core.serialization import fingerprint, sha256_bytes
from features.builder import FEATURE_BUILDER_VERSION, FEATURE_NAMES

MANIFEST_FILE: Final = "model-manifest.json"
EVENT_MODEL_FILE: Final = "event-model.joblib"
STATE_MODEL_FILE: Final = "state-model.joblib"
_PICKLE_LIBRARY: Final = "scikit-learn"


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    manifest: ModelManifest
    event_classifier: TurnClassifier
    state_classifier: TurnClassifier


def pack_estimator(estimator: Estimator) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    return buffer.getvalue()


def model_file(path: str, data: bytes) -> ModelFile:
    return ModelFile(path=path, sha256=sha256_bytes(data))


def write_bundle(
    directory: Path,
    manifest: ModelManifest,
    models: Mapping[str, bytes],
    documents: Mapping[str, object],
) -> None:
    for name, data in models.items():
        write_bytes_atomic(directory / name, data)
    for name, document in documents.items():
        write_json(directory / name, document)
    write_json(directory / MANIFEST_FILE, manifest.model_dump(mode="json"))


def load_bundle(directory: Path) -> LoadedBundle:
    manifest = _read_manifest(directory / MANIFEST_FILE)
    _check_compatible(manifest)
    _check_documents(directory, manifest)

    event = _load_estimator(directory, manifest.event_model)
    state = _load_estimator(directory, manifest.state_model)

    return LoadedBundle(
        manifest=manifest,
        event_classifier=TurnClassifier(
            manifest.feature_names,
            event,
            turn_threshold=manifest.event_turn_threshold,
        ),
        state_classifier=TurnClassifier(manifest.feature_names, state),
    )


def _read_manifest(path: Path) -> ModelManifest:
    try:
        return ModelManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8")), strict=False
        )
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ArtifactError(f"unreadable model manifest {path}: {exc}") from exc


def _check_compatible(manifest: ModelManifest) -> None:
    expected = {
        "artifact_kind": (manifest.artifact_kind, ARTIFACT_KIND),
        "bundle_format_version": (manifest.bundle_format_version, BUNDLE_FORMAT_VERSION),
        "window_length": (manifest.window_length, WINDOW_LENGTH),
        "feature_schema_version": (manifest.feature_schema_version, FEATURE_BUILDER_VERSION),
        "feature_names": (manifest.feature_names, FEATURE_NAMES),
        _PICKLE_LIBRARY: (manifest.runtime.packages.get(_PICKLE_LIBRARY), version(_PICKLE_LIBRARY)),
    }

    mismatched = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    if mismatched:
        raise ArtifactError(f"bundle is incompatible with this runtime: {mismatched}")


def _check_documents(directory: Path, manifest: ModelManifest) -> None:
    for name, checksum in manifest.documents.items():
        try:
            document = json.loads((directory / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"unreadable bundle document {name}: {exc}") from exc

        if fingerprint(document) != checksum:
            raise ArtifactError(f"{name} does not match its manifest checksum")


def _load_estimator(directory: Path, file: ModelFile) -> Estimator:
    path = directory / file.path
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"missing model file {path}") from exc

    if sha256_bytes(data) != file.sha256:
        raise ArtifactError(f"{file.path} does not match its manifest checksum")

    estimator: Estimator = joblib.load(io.BytesIO(data))
    return estimator
