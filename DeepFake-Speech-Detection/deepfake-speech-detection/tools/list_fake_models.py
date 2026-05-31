#!/usr/bin/env python3
"""
Count audio entries per `model_or_speaker` for entries where `label == "fake"`.

Supports:
- JSON array files (stream-parsed, so it can handle large arrays)
- JSONL files (one JSON object per line)

Usage:
  python tools/list_fake_models.py /path/to/labels.json
  python tools/list_fake_models.py /path/to/labels.json --label fake
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterator, Optional


def _skip_ws(buf: str, pos: int) -> int:
    n = len(buf)
    while pos < n and buf[pos].isspace():
        pos += 1
    return pos


def iter_json_array_items(path: str, *, read_size: int = 1024 * 1024) -> Iterator[Any]:
    """
    Incrementally parses a top-level JSON array: [ {...}, {...}, ... ]
    Yields each item (typically dicts).
    """
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    started = False
    eof = False

    with open(path, "r", encoding="utf-8") as f:
        while True:
            # Ensure we have enough buffer to keep parsing.
            if pos >= len(buf) - 4096 and not eof:
                chunk = f.read(read_size)
                if chunk:
                    buf = buf[pos:] + chunk
                    pos = 0
                else:
                    eof = True

            if eof and pos >= len(buf):
                break

            pos = _skip_ws(buf, pos)
            if not started:
                if pos >= len(buf):
                    continue
                if buf[pos] != "[":
                    raise ValueError("Not a JSON array file (does not start with '[').")
                started = True
                pos += 1
                continue

            pos = _skip_ws(buf, pos)
            if pos >= len(buf):
                continue

            ch = buf[pos]
            if ch == "]":
                return
            if ch == ",":
                pos += 1
                continue

            try:
                item, new_pos = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                # Likely incomplete buffer; read more.
                if eof:
                    raise
                chunk = f.read(read_size)
                if not chunk:
                    eof = True
                    continue
                buf = buf[pos:] + chunk
                pos = 0
                continue

            yield item
            pos = new_pos


def iter_jsonl_items(path: str) -> Iterator[Any]:
    """Parses a JSONL file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def iter_items_auto(path: str) -> Iterator[Any]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    first_non_ws: Optional[str] = None
    for c in head:
        if not c.isspace():
            first_non_ws = c
            break

    if first_non_ws == "[":
        yield from iter_json_array_items(path)
    else:
        yield from iter_jsonl_items(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print counts per model_or_speaker for entries where label matches (default: fake)."
    )
    parser.add_argument("path", help="Path to labels JSON array or JSONL file")
    parser.add_argument("--label", default="fake", help="Label value to filter by (default: fake)")
    args = parser.parse_args()

    wanted_label = args.label
    counts: Dict[str, int] = {}

    for item in iter_items_auto(args.path):
        if not isinstance(item, dict):
            continue
        d: Dict[str, Any] = item
        if d.get("label") != wanted_label:
            continue
        mos = d.get("model_or_speaker")
        if isinstance(mos, str) and mos.strip():
            counts[mos] = counts.get(mos, 0) + 1

    # Print "count<TAB>model" sorted by count desc then model asc for stability.
    for model, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{cnt}\t{model}")

    print(f"Total fake samples: {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

