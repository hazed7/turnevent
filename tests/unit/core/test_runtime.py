import platform
from pathlib import Path

from core.runtime import describe_runtime
from core.serialization import sha256_file


def test_reports_installed_versions_of_declared_dependencies(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["pydantic>=2", "PyYAML[extra]>=6 ; python_version >= \'3\'"]\n',
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")

    runtime = describe_runtime(tmp_path)

    assert runtime.python == platform.python_version()
    assert runtime.lockfile_sha256 == sha256_file(tmp_path / "uv.lock")
    assert list(runtime.packages) == ["PyYAML", "pydantic"]
    assert all(runtime.packages.values())
