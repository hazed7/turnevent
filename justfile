set shell := ["bash", "-euo", "pipefail", "-c"]
export PYTHONPATH := "src"

default:
    @just --list

install:
    uv sync --locked
    uv run pre-commit install

fmt:
    uv run ruff format .
    uv run ruff check --fix .

lint:
    uv run ruff format --check .
    uv run ruff check
    uv run lint-imports

typecheck:
    uv run mypy

test *args:
    uv run pytest --cov {{ args }}

check: lint typecheck test

pipeline:
    uv run python -m cli all

# Explicit synthetic demo; refuses an existing dataset.
demo:
    uv run python -m cli demo

# Forward selection/source/output options to the preparation script.
data-prepare *args:
    uv run python scripts/prepare_dataset.py {{ args }}

data-audit config="configs/real-data.yaml":
    uv run python -m cli --config {{ quote(config) }} audit

train-real config="configs/real-data.yaml":
    uv run python -m cli --config {{ quote(config) }} train

cli *args:
    uv run python -m cli {{ args }}
