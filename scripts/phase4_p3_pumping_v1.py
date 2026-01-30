#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_target(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    print(
        "NOTE: phase4_p3_pumping_v1 is an alias for phase3_p3_pumping_v4 (plan numbering fix)",
        file=sys.stderr,
    )
    target = _load_target("phase3_p3_pumping_v4", "phase3_p3_pumping_v4.py")
    if not hasattr(target, "main"):
        raise SystemExit("phase3_p3_pumping_v4.py does not expose main()")
    target.main()


if __name__ == "__main__":
    main()
