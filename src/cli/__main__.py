import argparse
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from adapter.publisher import Mode
from cli import commands
from cli.config import DEFAULT_CONFIG, PipelineConfig, load_config
from core.errors import TurnEventError


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        _run(args, config)
    except TurnEventError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


def _run(args: argparse.Namespace, config: PipelineConfig) -> None:
    handlers: dict[str, Callable[[], str]] = {
        "synthetic": lambda: f"dataset written to {commands.synthetic(config)}",
        "audit": lambda: f"data audit written to {commands.audit(config)}",
        "train": lambda: f"trained {commands.train(config).model_version}",
        "infer": lambda: _infer(args, config),
        "smoke": lambda: f"online report written to {commands.smoke(config, args.bundle)}",
        "all": lambda: f"pipeline finished: {commands.run_all(config).model_version}",
    }
    sys.stdout.write(handlers[args.command]() + "\n")


def _infer(args: argparse.Namespace, config: PipelineConfig) -> str:
    decisions = commands.infer(
        config, fixture=args.fixture, url=args.url, mode=Mode(args.mode), bundle=args.bundle
    )
    counts = Counter(d.outcome for d in decisions)
    return "\n".join(f"{outcome}: {count}" for outcome, count in sorted(counts.items()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cli", description="turn event pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("synthetic", help="generate the synthetic stand-in dataset")
    sub.add_parser("audit", help="data audit report, dataset manifest and label definition")
    sub.add_parser("train", help="train, evaluate, export the bundle and write reports")
    infer = sub.add_parser("infer", help="score one DS API envelope and publish signals")
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path, help="local JSON envelope")
    source.add_argument("--url", help="DS API endpoint returning an envelope")
    infer.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.DRY_RUN.value)
    infer.add_argument("--bundle", type=Path, help="bundle directory (default: latest)")
    smoke = sub.add_parser("smoke", help="run the online fixtures and write the online report")
    smoke.add_argument("--bundle", type=Path)
    sub.add_parser("all", help="synthetic -> audit -> train -> smoke")
    return parser


if __name__ == "__main__":
    sys.exit(main())
