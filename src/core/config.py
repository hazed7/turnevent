from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from core.errors import ConfigError


def load_yaml_config[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} must be a mapping, got {type(raw).__name__}")

    try:
        return model.model_validate(raw, strict=False)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc
