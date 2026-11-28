#!/usr/bin/env python3
"""Log metrics for the nightly/session checks: per-day counts from a shitbot.log COPY.

The failure modes that matter for clanker-channel quality are all single log lines:

  - LLM read timeouts (a reply died silently after the full request timeout)
  - max_tokens cap hits (a runaway generation truncated instead of timing out)
  - other LLM request failures, web_search failures
  - greeting-guard firings (regenerations and skipped posts)
  - logins (restarts) and tracebacks
  - per-call LLM timing lines, folded into per-day call counts and average wall time / prompt size
  - prompt-part size lines, folded into per-day average system-prompt chars (total, member memory, background chatter)

This tool counts them per day so a session can report deltas instead of re-grepping the log by hand.
It reads only the file it is pointed at, prints counts only (never log content), and writes nothing.

Usage: python scripts/log_metrics.py <path-to-shitbot.log> [--days N]
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d{3} ")
_LLM_CALL = re.compile(r"llm call: model=\S+ wall=([\d.]+)s prompt_tokens=(\d+|\?) completion_tokens=(\d+|\?)")
_PARTS = re.compile(r"prompt parts: total_chars=(\d+) persona_chars=(\d+) memory_chars=(\d+) quirks_chars=(\d+) chatter_chars=(\d+)")

# metric -> substrings that must ALL appear in the line
_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("llm_timeouts", ("xAI request failed", "Read timed out")),
    ("cap_hits", ("response hit the max_tokens cap",)),
    ("web_search_failures", ("web_search failed",)),
    ("guard_regenerations", ("greeting repeated skeleton",)),
    ("guard_skips", ("greeting retry repeated skeleton",)),
    ("guard_skips", ("greeting retry unusable",)),
    ("logins", ("Logged in as",)),
]


def analyze_log(lines: list[str]) -> OrderedDict[str, dict[str, float]]:
    """Return {day: {metric: count}} in first-seen day order. LLM failures that are not read
    timeouts are counted as llm_failures_other so the timeout share is visible. Completed-call
    timing lines fold into llm_calls plus average wall seconds and prompt tokens for the day."""
    days: OrderedDict[str, dict[str, float]] = OrderedDict()
    timing: dict[str, list[float]] = {}  # day -> [calls, wall_sum, prompt_sum, prompt_n]
    parts: dict[str, list[float]] = {}  # day -> [n, total_sum, memory_sum, chatter_sum]
    last_day: str | None = None
    for line in lines:
        m = _LINE.match(line)
        if not m:
            # Tracebacks are bare continuation lines of the preceding record; attribute them to its day.
            if last_day is not None and line.startswith("Traceback (most recent call last):"):
                day = days[last_day]
                day["tracebacks"] = day.get("tracebacks", 0) + 1
            continue
        last_day = m.group(1)
        day = days.setdefault(last_day, {})
        part = _PARTS.search(line)
        if part:
            acc = parts.setdefault(last_day, [0, 0.0, 0.0, 0.0])
            acc[0] += 1
            acc[1] += int(part.group(1))
            acc[2] += int(part.group(3))
            acc[3] += int(part.group(5))
            continue
        call = _LLM_CALL.search(line)
        if call:
            acc = timing.setdefault(last_day, [0, 0.0, 0.0, 0])
            acc[0] += 1
            acc[1] += float(call.group(1))
            if call.group(2) != "?":
                acc[2] += int(call.group(2))
                acc[3] += 1
            continue
        matched = False
        for metric, subs in _PATTERNS:
            if all(s in line for s in subs):
                day[metric] = day.get(metric, 0) + 1
                matched = True
                if metric == "llm_timeouts":
                    break  # a timeout line is not also an "other" failure
        if not matched and "xAI request failed" in line:
            day["llm_failures_other"] = day.get("llm_failures_other", 0) + 1
    for day_name, (calls, wall_sum, prompt_sum, prompt_n) in timing.items():
        day = days[day_name]
        day["llm_calls"] = int(calls)
        day["llm_wall_avg_s"] = round(wall_sum / calls, 1)
        if prompt_n:
            day["llm_prompt_tokens_avg"] = round(prompt_sum / prompt_n)
    for day_name, (n, total_sum, memory_sum, chatter_sum) in parts.items():
        day = days[day_name]
        day["prompt_parts"] = int(n)
        day["sys_chars_avg"] = round(total_sum / n)
        day["memory_chars_avg"] = round(memory_sum / n)
        day["chatter_chars_avg"] = round(chatter_sum / n)
    return days


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    days_n = 14
    if "--days" in argv:
        days_n = int(argv[argv.index("--days") + 1])
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    days = analyze_log(lines)
    metrics = ["llm_timeouts", "cap_hits", "llm_failures_other", "web_search_failures",
               "guard_regenerations", "guard_skips", "logins", "tracebacks",
               "llm_calls", "llm_wall_avg_s", "llm_prompt_tokens_avg",
               "prompt_parts", "sys_chars_avg", "memory_chars_avg", "chatter_chars_avg"]
    print(f"{'day':<12}" + "".join(f"{m:>22}" for m in metrics))
    for day in list(days)[-days_n:]:
        counts = days[day]
        print(f"{day:<12}" + "".join(f"{counts.get(m, 0):>22}" for m in metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
