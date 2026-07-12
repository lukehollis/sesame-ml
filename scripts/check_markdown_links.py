#!/usr/bin/env python3
"""Fail when repository Markdown points to a missing local file or render report."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    markdown = sorted(
        {
            *root.glob("*.md"),
            *root.glob("docs/**/*.md"),
            *root.glob("integrations/**/*.md"),
        }
    )
    errors: list[str] = []
    checked = 0
    for source in markdown:
        for raw_target in LINK.findall(source.read_text(encoding="utf-8")):
            target = raw_target.strip().split()[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = unquote(target.split("#", 1)[0])
            if not local:
                continue
            checked += 1
            if not (source.parent / local).resolve().exists():
                errors.append(f"{source.relative_to(root)}: missing {target}")

    for report in sorted((root / "docs/media/reports").glob("*/report.json")):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            episodes = data["episodes"]
            if len(episodes) != 1:
                raise ValueError("public render report must contain exactly one episode")
            video = root / episodes[0]["video"]
            if not video.is_file():
                raise ValueError(f"referenced video is missing: {video}")
            if not report.with_name("episodes.csv").is_file():
                raise ValueError("episodes.csv is missing")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{report.relative_to(root)}: {error}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"checked {checked} local links across {len(markdown)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
