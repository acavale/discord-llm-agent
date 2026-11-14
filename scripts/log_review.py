#!/usr/bin/env python3
"""Review recent agent activity from the JSONL message archive.

Two things it answers:
  1) Is the agent responding to what people are ACTUALLY talking about? (prints an
     interleaved transcript per channel, agent lines marked)
  2) Is it fabricating links/media? (flags every URL or media claim in agent output --
     a local model has no way to know real video/social URLs, so any specific link it
     emits is almost certainly invented)

Usage:  python scripts/log_review.py [hours] <path/to/guild_<id>.jsonl>
Channel names are read from the storage root's config.yaml when available.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

if len(sys.argv) < 3:
    sys.exit("usage: log_review.py [hours] <archive.jsonl>")
ARCHIVE = Path(sys.argv[2])


def _channel_names() -> dict[str, str]:
    """Channel id -> name from the config.yaml next to the archive, if present."""
    try:
        import yaml
        cfg = yaml.safe_load((ARCHIVE.parent / "config.yaml").read_text(encoding="utf-8")) or {}
        return {str(k): (v or {}).get("name", str(k)) for k, v in (cfg.get("channels") or {}).items()}
    except Exception:  # noqa: BLE001 -- names are cosmetic; fall back to raw ids
        return {}


CHANNELS = _channel_names()

URL_RE = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|gg|tv|io|ly)/\S*", re.I)
MEDIA_RE = re.compile(
    r"\b(tiktok|reel|reels|insta(?:gram)?|youtu\.?be|youtube|shorts|twitch|clip)\b", re.I
)


def ascii_safe(s: str) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 else "." for ch in s)


def main() -> int:
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    cutoff = datetime.now() - timedelta(hours=hours)

    rows = []
    with ARCHIVE.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ts = r.get("timestamp") or ""
            try:
                when = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if when >= cutoff:
                r["_when"] = when
                rows.append(r)

    rows.sort(key=lambda r: r["_when"])
    print(f"=== {len(rows)} messages in the last {hours}h ===\n")

    # ---- per-channel transcript ----
    by_ch: dict[str, list] = {}
    for r in rows:
        by_ch.setdefault(str(r.get("channel_id")), []).append(r)

    for cid, msgs in by_ch.items():
        name = CHANNELS.get(cid, f"channel {cid}")
        bots = sum(1 for m in msgs if (m.get("username") or "") == "agent")
        print("=" * 78)
        print(f"#{name}   {len(msgs)} msgs ({bots} from bot, {len(msgs) - bots} human)")
        print("=" * 78)
        for m in msgs[-40:]:
            who = m.get("username") or "?"
            tag = "BOT >>" if who == "agent" else "      "
            txt = ascii_safe((m.get("message") or "").replace("\n", " / "))
            print(f"{m['_when'].strftime('%m-%d %H:%M')} {tag} {who[:12]:12} | {txt[:230]}")
        print()

    # ---- fabricated-link audit ----
    print("=" * 78)
    print("LINK / MEDIA AUDIT (bot messages only)")
    print("=" * 78)
    flagged = 0
    for r in rows:
        if (r.get("username") or "") != "agent":
            continue
        txt = r.get("message") or ""
        urls = URL_RE.findall(txt)
        media = MEDIA_RE.findall(txt)
        if urls or media:
            flagged += 1
            ch = CHANNELS.get(str(r.get("channel_id")), r.get("channel_id"))
            print(f"\n[{r['_when'].strftime('%m-%d %H:%M')}] #{ch}")
            print("  " + ascii_safe(txt.replace("\n", " / "))[:300])
            if urls:
                print("  URLS: " + ", ".join(ascii_safe(u)[:80] for u in urls[:4]))
            if media:
                print("  MEDIA WORDS: " + ", ".join(sorted(set(w.lower() for w in media))))
    if not flagged:
        print("  (none found in this window)")
    else:
        print(f"\n  {flagged} bot messages referenced links/media.")

    # ---- repetition check ----
    print("\n" + "=" * 78)
    print("REPETITION (most repeated bot phrases, 6-word runs)")
    print("=" * 78)
    runs: Counter[str] = Counter()
    for r in rows:
        if (r.get("username") or "") != "agent":
            continue
        w = re.sub(r"[^a-z0-9 ]+", " ", (r.get("message") or "").lower()).split()
        for i in range(max(0, len(w) - 5)):
            runs[" ".join(w[i:i + 6])] += 1
    for phrase, n in runs.most_common(8):
        if n > 1:
            print(f"  {n}x  {phrase}")
    if not any(n > 1 for _, n in runs.most_common(8)):
        print("  (no repeated 6-word runs)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
