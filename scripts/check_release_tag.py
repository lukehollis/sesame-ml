#!/usr/bin/env python3
"""Require a release tag to match the PEP 621 project version exactly."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    args = parser.parse_args()
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    expected = f"v{version}"
    if args.tag != expected:
        parser.error(f"release tag {args.tag!r} must equal {expected!r}")
    print(f"release tag {args.tag} matches project version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
