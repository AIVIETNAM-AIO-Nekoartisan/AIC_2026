from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .runner import (
    RunContext,
    default_stage_handler,
    iter_dataset_jobs,
    run_full,
    run_pilot,
    run_preflight,
)
from .state import StateStore


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/Video AIC"))
    parser.add_argument("--work-root", type=Path, default=Path("/content/aic25-work"))
    parser.add_argument(
        "--bundle-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aic_colab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    subparsers.add_parser("preflight", parents=[common])
    pilot = subparsers.add_parser("pilot", parents=[common])
    pilot.add_argument("--limit", type=int, default=1000)
    full = subparsers.add_parser("full", parents=[common])
    full.add_argument("--deadline-seconds", type=int, default=31500)
    subparsers.add_parser("status", parents=[common])
    return parser


def _context(args: argparse.Namespace) -> RunContext:
    config_path = args.config or args.bundle_root / "config" / "aic25-b1.json"
    config = load_config(config_path)
    version_path = args.bundle_root / "VERSION"
    version = version_path.read_text().strip() if version_path.is_file() else "1"
    state_store = StateStore(
        args.drive_root / "AIC25-Colab" / "runtime" / "state.json",
        config.dataset_id,
        version,
    )
    return RunContext(
        config=config,
        drive_root=args.drive_root,
        work_root=args.work_root,
        bundle_root=args.bundle_root,
        state_store=state_store,
        jobs=iter_dataset_jobs,
        stage_handler=default_stage_handler,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = _context(args)
    try:
        if args.command == "status":
            print(json.dumps(ctx.state_store.load(), indent=2, sort_keys=True))
            return 0
        if args.command == "preflight":
            report = run_preflight(ctx)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ready"] else 2
        if args.command == "pilot":
            report = run_pilot(ctx, args.limit)
        else:
            preflight = run_preflight(ctx)
            if not preflight["ready"]:
                print("preflight failed", file=sys.stderr)
                return 2
            pilot_path = args.drive_root / "AIC25-Colab" / "pilot" / "report.json"
            if not pilot_path.is_file():
                print("pilot report is missing", file=sys.stderr)
                return 2
            pilot = json.loads(pilot_path.read_text())
            if (
                pilot.get("status") != "passed"
                or pilot.get("bundle_version") != ctx.state_store.bundle_version
            ):
                print("pilot gate has not passed for this bundle", file=sys.stderr)
                return 2
            report = run_full(ctx, args.deadline_seconds)
        print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
