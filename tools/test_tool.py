#!/usr/bin/env python3
"""Dry-run a Tools.json config without building the app.

    python tools/test_tool.py "what is bitcoin trading at" [path/to/Tools.json]

Prints the matched tool and the fact sentence that would be folded into the
prompt. Mirrors the URL templating, field extraction, derived fields and
fail-closed error handling of ToolBox.kt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


def resolve(node, path):
    for key in path:
        if node is None:
            return None
        if key == "*":
            node = next(iter(node.values()), None) if isinstance(node, dict) else (node[0] if node else None)
        elif isinstance(node, list) and key.isdigit():
            i = int(key)
            node = node[i] if 0 <= i < len(node) else None
        elif isinstance(node, dict):
            node = node.get(key)
        else:
            return None
    return node


def evaluate(expr: str, nums: dict[str, float]) -> float | None:
    tokens = [t for t in re.split(r"([+\-*/])", expr) if t.strip()]
    if not tokens:
        return None
    try:
        acc = nums.get(tokens[0].strip(), float(tokens[0]))
    except ValueError:
        return None
    i = 1
    while i + 1 < len(tokens):
        op, name = tokens[i].strip(), tokens[i + 1].strip()
        try:
            v = nums.get(name, float(name))
        except ValueError:
            return None
        if op == "/":
            if v == 0:
                return None
            acc /= v
        elif op == "*":
            acc *= v
        elif op == "+":
            acc += v
        elif op == "-":
            acc -= v
        else:
            return None
        i += 2
    return acc


def run_tool(spec: dict, query: str) -> str:
    url = spec["url"]
    if spec.get("windowDays"):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=spec["windowDays"])
        url = url.replace("{startDate}", start.strftime("%Y-%m-%d"))
        url = url.replace("{endDate}", end.strftime("%Y-%m-%d"))
    if "{query}" in url:
        topic = query.lower()
        for t in spec.get("triggers", []):
            if topic.startswith(t.lower()):
                topic = topic[len(t):]
        topic = topic.strip(" ?.!,")
        url = url.replace("{query}", urllib.parse.quote(topic))

    req = urllib.request.Request(url, headers=spec.get("headers") or {})
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.load(resp)

    paths = dict(spec.get("fields") or {})
    if spec.get("jsonPath"):
        paths["value"] = spec["jsonPath"]
    if not paths:
        raise RuntimeError("tool response did not contain the expected field")

    text: dict[str, str] = {}
    nums: dict[str, float] = {}
    for name, path in paths.items():
        node = resolve(payload, path)
        if node is None:
            raise RuntimeError("tool response did not contain the expected field")
        raw = str(node)
        limit = spec.get("maxChars")
        if limit and len(raw) > limit:
            cut = raw[:limit]
            raw = cut[: cut.rfind(".") + 1] or cut
        text[name] = raw
        try:
            nums[name] = float(raw.replace(",", ""))
        except ValueError:
            pass
    for field in spec.get("derived") or []:
        v = evaluate(field["expr"], nums)
        if v is None:
            continue
        nums[field["name"]] = v
        text[field["name"]] = f"{v:.1f}" if v < 100 else f"{v:,.0f}"

    out = spec["factTemplate"]
    for name, value in text.items():
        out = out.replace("{" + name + "}", value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("config", nargs="?", default="app/src/main/assets/Tools.json")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.exists():
        print(f"no tool config at {path} (copy Tools.example.json)", file=sys.stderr)
        return 2
    specs = json.loads(path.read_text(encoding="utf-8"))

    q = args.query.lower()
    match = next((s for s in specs if any(t.lower() in q for t in s.get("triggers", []))), None)
    if match is None:
        print("no tool matched; the prompt would go straight to the model")
        return 0

    print(f"matched tool: {match['name']}")
    try:
        fact = run_tool(match, args.query)
    except Exception as e:  # fail closed, like LlmRunner
        print(f"FAILED: {e}\nthe app would NOT ask the model (fail closed)")
        return 1
    print(f"fact: {fact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
