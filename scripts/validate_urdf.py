#!/usr/bin/env python3
"""Validate the packaged URDF structure and numerical equivalence to MuJoCo."""

from __future__ import annotations

import argparse
import json

from sesame_ml.urdf import assert_urdf_equivalent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int, default=256, help="random poses in addition to stand"
    )
    parser.add_argument("--seed", type=int, default=20260712)
    arguments = parser.parse_args()
    report = assert_urdf_equivalent(samples=arguments.samples, seed=arguments.seed)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
