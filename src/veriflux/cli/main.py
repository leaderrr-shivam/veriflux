"""Command-line interface for Veriflux.

Provides commands to run test suites, manage regression baselines, and initialize configs.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from veriflux.__version__ import __version__
from veriflux.cli.formatters import TerminalFormatter
from veriflux.regression.baseline import BaselineManager


def _cmd_init(args: argparse.Namespace, fmt: TerminalFormatter) -> int:
    config_path = "veriflux.toml"
    if os.path.exists(config_path):
        print(fmt.warning(f"Configuration file '{config_path}' already exists."))
        return 0

    default_toml = """# Veriflux Configuration
[provider]
model = "gpt-4o"
timeout = 30.0

[thresholds]
groundedness = 0.75
faithfulness = 0.70
toxicity = 0.50

[cache]
enabled = true
max_size = 4096
ttl = 3600

[guardrails]
block_pii = true
block_prompt_injection = true
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(default_toml)

    print(fmt.success(f"Initialized Veriflux configuration in '{config_path}'"))
    return 0


def _cmd_baseline(args: argparse.Namespace, fmt: TerminalFormatter) -> int:
    bm = BaselineManager()
    action = args.baseline_action

    if action == "save":
        # Mock/sample metric save for CLI demonstration
        name = args.name or "latest"
        metrics = {"groundedness": 0.88, "latency_ms": 420.0, "cost_usd": 0.002}
        bm.save(name, metrics)
        print(fmt.success(f"Saved baseline snapshot '{name}'"))
        return 0

    elif action == "compare":
        name = args.name or "latest"
        try:
            curr = {"groundedness": 0.86, "latency_ms": 450.0, "cost_usd": 0.002}
            res = bm.compare(name, curr)
            print(fmt.header(f"Baseline Comparison: {name}"))
            for reg in res.regressions:
                print(fmt.error(reg))
            for imp in res.improvements:
                print(fmt.success(imp))
            for unc in res.unchanged:
                print(fmt.warning(unc))
            return 1 if res.has_regressions else 0
        except FileNotFoundError:
            print(fmt.error(f"Baseline '{name}' not found."))
            return 1

    return 0


def _cmd_version(args: argparse.Namespace, fmt: TerminalFormatter) -> int:
    print(f"[Veriflux] v{__version__} - Truth flows through every layer.")
    return 0


def _cmd_run(args: argparse.Namespace, fmt: TerminalFormatter) -> int:
    target = args.path or "tests"
    print(fmt.header(f"Running Veriflux Evaluation Test Suite on '{target}'"))
    # Delegate to pytest
    try:
        import pytest
        ret = pytest.main([target] + (["-v"] if args.verbose else []))
        return int(ret)
    except ImportError:
        print(fmt.error("pytest is required to execute test runs. Install with: pip install pytest"))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veriflux",
        description="Veriflux -- The definitive AI/ML testing and evaluation library.",
    )
    parser.add_argument("-V", "--version", action="store_true", help="Display version and exit.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute.")

    # run
    p_run = subparsers.add_parser("run", help="Execute evaluation test suite.")
    p_run.add_argument("path", nargs="?", default="tests", help="Path to tests or dataset.")
    p_run.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output.")

    # init
    subparsers.add_parser("init", help="Initialize veriflux.toml in current directory.")

    # baseline
    p_base = subparsers.add_parser("baseline", help="Manage and compare regression baselines.")
    p_base.add_argument("baseline_action", choices=["save", "compare"], help="Baseline action.")
    p_base.add_argument("name", nargs="?", default="latest", help="Baseline name.")

    # version
    subparsers.add_parser("version", help="Print Veriflux version.")

    return parser


def cli_entry(argv: Optional[List[str]] = None) -> int:
    """CLI application entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    fmt = TerminalFormatter()

    if args.version:
        return _cmd_version(args, fmt)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "init":
        return _cmd_init(args, fmt)
    elif args.command == "baseline":
        return _cmd_baseline(args, fmt)
    elif args.command == "version":
        return _cmd_version(args, fmt)
    elif args.command == "run":
        return _cmd_run(args, fmt)

    return 0


if __name__ == "__main__":
    sys.exit(cli_entry())
