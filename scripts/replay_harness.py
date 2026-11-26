#!/usr/bin/env python3
"""Replay harness: feed a recorded message archive through the bot's routing rules, offline.

This is a check tool, not part of the bot. It reads COPIES of the archive and config, never writes
to live state, and never calls a model or the network. It exists so the response-coverage,
gate-cause, and direct-address audits produce comparable numbers instead of impressions:

- every human message in a configured (chime) channel is classified the way the client routes it:
  a direct address (@-mention of the bot, or a name-drop on one of the configured aliases) or
  ambient traffic;
- the outcome is what actually happened: did a bot message land in the same channel within the
  follow-up window? (A scheduled post inside the window can coincide; that is the same caveat the
  manual coverage audit always carried.)
- for a missed direct address, a mirror of the rhythm budget (same capacity/refill math as
  discord_agent.gate.RhythmBudget, driven by archive timestamps instead of the wall clock) says
  whether the hourly bucket was spent at that moment -- the one structural cause the archive alone
  can prove. Any other miss is reported as unexplained (model failure, timeout, or downtime; the
  log decides, not this tool).
- ambient messages in non-"all" channels are scored with the real Gate against a replayed
  transcript, so gate rejections are counted with the client's own logic.

Known limits: replies-to-the-bot are not recorded in the archive, so a reply that never names the
bot is counted as ambient; the budget mirror treats a bot send as budget-consuming when a direct
address arrived in the channel shortly before it, which approximates the debounce merging in the
client.

Output is counts only: no message text, and users are keyed by a short hash, never by name or id.

    python scripts/replay_harness.py --archive guild.jsonl --config config.yaml [--days 7]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from discord_agent.gate import Gate  # noqa: E402

FOLLOWUP_S = 600.0  # a bot post within 10 minutes counts as the follow-up (the coverage audit's window)
DIRECTED_SEND_WINDOW_S = 120.0  # a send this soon after a direct address is treated as its reply


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _user_key(user_id: str) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:8]


@dataclass
class _Entry:
    ts: float
    is_bot: bool = False


class _ReplayTranscripts:
    """The slice of context.Transcripts the Gate reads, replayed on archive time."""

    def __init__(self) -> None:
        self._buf: dict[str, list[_Entry]] = {}
        self._now = 0.0

    def add(self, cid: str, ts: float, is_bot: bool) -> None:
        self._buf.setdefault(cid, []).append(_Entry(ts=ts, is_bot=is_bot))
        self._now = ts

    def recent(self, channel_id: str, n: int | None = None) -> list[_Entry]:
        items = list(self._buf.get(channel_id, []))
        return items[-n:] if n else items

    def velocity(self, channel_id: str, window_s: float = 90.0) -> int:
        cutoff = self._now - window_s
        return sum(1 for e in self._buf.get(channel_id, []) if e.ts >= cutoff)


@dataclass
class _Bucket:
    """RhythmBudget's math on archive time: continuous refill, 1 token per consuming send."""

    capacity: float
    tokens: float
    last_refill: float

    def refill(self, ts: float) -> None:
        elapsed = max(0.0, ts - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * (self.capacity / 3600.0))
        self.last_refill = ts

    def has_budget(self, ts: float) -> bool:
        self.refill(ts)
        return self.tokens >= 1.0

    def consume(self, ts: float) -> None:
        self.refill(ts)
        self.tokens = max(0.0, self.tokens - 1.0)


@dataclass
class _ChannelStats:
    human: int = 0
    directed: int = 0
    directed_answered: int = 0
    directed_missed: int = 0
    miss_budget_spent: int = 0
    miss_unexplained: int = 0
    ambient: int = 0
    ambient_answered: int = 0
    ambient_gate_rejected: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class _ChannelState:
    stats: _ChannelStats = field(default_factory=_ChannelStats)
    bucket: _Bucket | None = None
    last_directed_ts: float = float("-inf")
    last_send_ts: float = float("-inf")


def analyze(rows: list[dict], cfg: dict, bot_username: str = "agent",
            days: float | None = None) -> dict:
    """Classify every human message in the configured channels and compare with what happened.

    rows are archive dicts (timestamp, channel_id, user_id, username, message); cfg is the parsed
    config.yaml. Pure: inputs are not mutated and nothing is written anywhere.
    """
    rows = sorted(rows, key=lambda r: r["timestamp"])
    if not rows:
        return {"channels": {}, "readonly_human": 0, "users_with_repeated_direct_misses": {}}
    if days is not None:
        end = _parse_ts(rows[-1]["timestamp"])
        start = end - timedelta(days=days)
        rows = [r for r in rows if _parse_ts(r["timestamp"]) >= start]

    bot_ids = {str(r["user_id"]) for r in rows if r.get("username") == bot_username}
    channels_cfg: dict = cfg.get("channels") or {}
    chime_cfg: dict = cfg.get("chime_in") or {}
    aliases = [a.lower() for a in (chime_cfg.get("name_aliases") or [])]
    hourly = float(chime_cfg.get("per_channel_hourly_budget", 6))
    gate = Gate(floor=float(chime_cfg.get("gate_floor", 0.5)))

    transcripts = _ReplayTranscripts()
    states: dict[str, _ChannelState] = {}
    bot_ts_by_channel: dict[str, list[float]] = {}
    for r in rows:
        if str(r.get("user_id")) in bot_ids:
            bot_ts_by_channel.setdefault(str(r["channel_id"]), []).append(
                _parse_ts(r["timestamp"]).timestamp())

    def answered(cid: str, ts: float) -> bool:
        return any(ts < b <= ts + FOLLOWUP_S for b in bot_ts_by_channel.get(cid, []))

    misses_by_user: dict[str, int] = {}
    readonly_human = 0

    for r in rows:
        cid = str(r["channel_id"])
        ts = _parse_ts(r["timestamp"]).timestamp()
        is_bot = str(r.get("user_id")) in bot_ids
        transcripts.add(cid, ts, is_bot)
        if is_bot:
            st = states.get(cid)
            if st is not None and st.bucket is not None:
                chime = channels_cfg.get(cid, {}).get("chime_in", "off")
                directed_send = ts - st.last_directed_ts <= DIRECTED_SEND_WINDOW_S \
                    and st.last_directed_ts > st.last_send_ts
                if directed_send or chime != "all":
                    st.bucket.consume(ts)
            if st is not None:
                st.last_send_ts = ts
            continue
        if cid not in channels_cfg:
            readonly_human += 1
            continue
        st = states.setdefault(cid, _ChannelState())
        if st.bucket is None:
            st.bucket = _Bucket(capacity=hourly, tokens=hourly, last_refill=ts)
        st.stats.human += 1
        text = (r.get("message") or "")
        low = text.lower()
        directed = any(f"<@{b}>" in text or f"<@!{b}>" in text for b in bot_ids) \
            or any(alias in low for alias in aliases)
        if directed:
            st.stats.directed += 1
            st.last_directed_ts = ts
            if answered(cid, ts):
                st.stats.directed_answered += 1
            else:
                st.stats.directed_missed += 1
                if st.bucket.has_budget(ts):
                    st.stats.miss_unexplained += 1
                else:
                    st.stats.miss_budget_spent += 1
                key = _user_key(r.get("user_id", ""))
                misses_by_user[key] = misses_by_user.get(key, 0) + 1
            continue
        st.stats.ambient += 1
        chime = channels_cfg.get(cid, {}).get("chime_in", "off")
        if chime != "all":
            floor = 0.75 if chime == "limited" else None
            ok, _reason = gate.should_consider(text, transcripts, cid, floor=floor)
            if not ok:
                st.stats.ambient_gate_rejected += 1
                continue
        if answered(cid, ts):
            st.stats.ambient_answered += 1

    out_channels = {}
    for cid, st in states.items():
        name = channels_cfg.get(cid, {}).get("name", cid)
        out_channels[name] = st.stats.as_dict()
    return {
        "channels": out_channels,
        "readonly_human": readonly_human,
        "users_with_repeated_direct_misses": {k: v for k, v in misses_by_user.items() if v >= 2},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay an archive copy through the routing rules.")
    ap.add_argument("--archive", required=True, help="path to a COPY of a guild archive .jsonl")
    ap.add_argument("--config", required=True, help="path to a COPY of config.yaml")
    ap.add_argument("--days", type=float, default=7.0, help="window ending at the archive's last row")
    ap.add_argument("--bot-username", default="agent",
                    help="archive username the bot posts under (default: agent)")
    ap.add_argument("--json-out", help="also write the counts as JSON to this path")
    args = ap.parse_args()

    import yaml

    rows = [json.loads(line) for line in Path(args.archive).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = analyze(rows, cfg, bot_username=args.bot_username, days=args.days)

    totals = _ChannelStats()
    for name, s in sorted(result["channels"].items()):
        print(f"{name}: human={s['human']} directed={s['directed']} "
              f"answered={s['directed_answered']} missed={s['directed_missed']} "
              f"(budget-spent={s['miss_budget_spent']}, unexplained={s['miss_unexplained']}) "
              f"ambient={s['ambient']} ambient-followed={s['ambient_answered']} "
              f"gate-rejected={s['ambient_gate_rejected']}")
        for k, v in s.items():
            setattr(totals, k, getattr(totals, k) + v)
    t = totals.as_dict()
    print(f"TOTAL: human={t['human']} directed={t['directed']} answered={t['directed_answered']} "
          f"missed={t['directed_missed']} (budget-spent={t['miss_budget_spent']}, "
          f"unexplained={t['miss_unexplained']}) ambient={t['ambient']} "
          f"ambient-followed={t['ambient_answered']} gate-rejected={t['ambient_gate_rejected']}")
    print(f"read-only channel human messages: {result['readonly_human']}")
    rep = result["users_with_repeated_direct_misses"]
    print(f"users (hashed) with 2+ missed direct addresses: {rep if rep else 'none'}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
