from pathlib import Path

import pytest

from artifact.bundle import LoadedBundle, load_bundle


@pytest.fixture(scope="session")
def bundle(trained_bundle_dir: Path) -> LoadedBundle:
    return load_bundle(trained_bundle_dir)
