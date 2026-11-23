"""When-to-speak gate (cheap, no model call) and rhythm budget (structural anti-spam).

The gate scores whether an ambient (non-addressed) message is worth a model call. The budget caps
spontaneous messages per channel and prevents the bot from talking over itself. Numbers come from config
and are meant to be tuned against real traffic later.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_QUESTION_WORDS = re.compile(r"\b(who|what|when|where|why|how|which|anyone|should|is it|does)\b", re.I)
_LOW_EFFORT = re.compile(r"^(lol|lmao|k|ok|yeah|yea|nah|same|fr|true|w|l|\+|\.)+$", re.I)


@dataclass
class _Bucket:
    tokens: float
    last_refill: float
    last_speak: float = 0.0


class RhythmBudget:
    """Token bucket per channel + post-speak cooldown + no-double-message rule."""

    def __init__(self, hourly_budget: int, post_speak_cooldown_s: float) -> None:
        self.capacity = float(hourly_budget)
        self.refill_per_s = hourly_budget / 3600.0
        self.cooldown = post_speak_cooldown_s
        self._buckets: dict[str, _Bucket] = {}

    def _bucket(self, cid: str) -> _Bucket:
        b = self._buckets.get(cid)
        now = time.time()
        if b is None:
            b = _Bucket(tokens=self.capacity, last_refill=now)
            self._buckets[cid] = b
        else:
            elapsed = now - b.last_refill
            b.tokens = min(self.capacity, b.tokens + elapsed * self.refill_per_s)
            b.last_refill = now
        return b

    def can_speak(self, cid: str) -> bool:
        b = self._bucket(cid)
        if time.time() - b.last_speak < self.cooldown:
            return False
        return b.tokens >= 1.0

    def has_budget(self, cid: str) -> bool:
        """Token check only, without the post-speak cooldown. Directed replies (a mention, a name-drop,
        an answer to the bot's riddle) often land seconds after the bot's own last post, and the
        cooldown in can_speak would eat exactly those; the hourly bucket alone is the spam cap there."""
        return self._bucket(cid).tokens >= 1.0

    def record_speak(self, cid: str) -> None:
        b = self._bucket(cid)
        b.tokens = max(0.0, b.tokens - 1.0)
        b.last_speak = time.time()

    def reset(self, cid: str | None = None) -> None:
        if cid is None:
            self._buckets.clear()
        else:
            self._buckets.pop(cid, None)


class Gate:
    """Lenient on purpose. The cheap gate's only job is to skip a model call on obvious non-starters
    (pure noise, or talking over itself). The real filtering is the model's own reluctance (it can stay
    quiet) plus the rhythm budget. Name-drops are handled upstream in the client (they fire in any
    channel), so the gate here is pure ambient salience.
    """

    def __init__(self, floor: float = 0.5) -> None:
        self.floor = floor

    def score(self, content: str, transcripts, channel_id: str) -> tuple[float, str]:
        text = (content or "").strip()
        if not text:
            return 0.0, "empty"

        # Never speak twice in a row without a human between (structural).
        recent = transcripts.recent(channel_id, 1)
        if recent and recent[0].is_bot:
            return 0.0, "would-double-up"

        # Pure low-effort reaction or a couple characters: not worth a model call.
        if _LOW_EFFORT.match(text) or len(text) < 3:
            return 0.0, "low-effort"

        # Everything else passes by default; the model decides whether to actually speak.
        score = 0.55
        reasons = ["baseline"]
        if text.endswith("?") or _QUESTION_WORDS.search(text):
            score += 0.15
            reasons.append("question")
        if transcripts.velocity(channel_id) <= 1:
            score += 0.1
            reasons.append("lull")
        return score, ",".join(reasons)

    def should_consider(self, content: str, transcripts, channel_id: str, floor: float | None = None):
        score, reason = self.score(content, transcripts, channel_id)
        f = self.floor if floor is None else floor
        return (score >= f), f"{reason} (score={score:.2f}, floor={f:.2f})"
