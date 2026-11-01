"""Run with PYTHONPATH=src, or use the just data-prepare recipe."""

import argparse
import json
from datetime import date
from pathlib import Path

from core.errors import DataContractError
from dataset.preparation import CHECKSUMS, Selection, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify frozen CSVs and prepare model-input Parquet"
    )
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/prepared/marketscreener-2026-10-v1.parquet")
    )
    parser.add_argument("--checksums", type=Path, default=CHECKSUMS)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--symbols", nargs="+", default=[])
    args = parser.parse_args()
    try:
        result = prepare_dataset(
            args.source,
            args.output,
            Selection(args.start_date, args.end_date, tuple(args.symbols)),
            args.checksums,
        )
    except (DataContractError, OSError, ValueError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps({key: value for key, value in result.items() if key != "inputs"}, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
