import platform
import tomllib
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement

from core.frozen import FrozenModel
from core.serialization import sha256_file


class RuntimeInfo(FrozenModel):
    python: str
    lockfile_sha256: str
    packages: Mapping[str, str]


def describe_runtime(project_root: Path) -> RuntimeInfo:
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(Requirement(spec).name for spec in pyproject["project"]["dependencies"])

    return RuntimeInfo(
        python=platform.python_version(),
        lockfile_sha256=sha256_file(project_root / "uv.lock"),
        packages={name: version(name) for name in names},
    )
