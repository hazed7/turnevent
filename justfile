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

demo:
    uv run python -m cli demo

[positional-arguments]
data-prepare *args:
    uv run python scripts/prepare_dataset.py "$@"

data-audit config="configs/real-data.yaml":
    uv run python -m cli --config {{ quote(config) }} audit

train-real config="configs/real-data.yaml":
    uv run python -m cli --config {{ quote(config) }} train

cli *args:
    uv run python -m cli {{ args }}
