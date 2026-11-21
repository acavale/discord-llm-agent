"""Scheduler: fires time-based and presence-based intents into the one agent.

Time-based: a list of candidate times, one chosen at random per day, gated by weekday and once-per-day.
Presence-based ("condition_based"): fires on a presence transition for matching users -- either coming
online, or (optionally) starting/switching a rich-presence activity (a game, Spotify, a stream) that matches
per-schedule filters (activity type, name substring, platform). Cooldown + optional probability, and yields
if the target channel is mid-conversation. All schedules live in scheduler.json and just describe *what* to
post; the agent decides *how*.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import random
from collections.abc import Awaitable, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Activity types that count as a "real" presence change always worth a post (starting or switching a game,
# stream, or match), vs low-signal noise (a bare login, a custom status, background music). A real activity
# start/switch fires every time; a plain come-online is gated by the long per-user greeting cooldown.
_REAL_ACTIVITY_TYPES = {"playing", "streaming", "competing"}


def _is_activity_change(ch: dict) -> bool:
    """True if this tick's change includes a real activity start/switch (game/stream/match), as opposed to a
    plain come-online, an activity stop, or low-signal noise. Stops never appear here because new_acts only
    holds additions, so closing a game produces no new activity."""
    return any(t in _REAL_ACTIVITY_TYPES for (t, _n) in ch.get("new_acts", ()))


class Scheduler:
    def __init__(
        self,
        scheduler_file: Path,
        on_intent: Callable[[dict, dict | None], Awaitable[None]],
        is_channel_busy: Callable[[str], bool] | None = None,
        tick_seconds: int = 30,
        greet_probability: float = 1.0,
    ) -> None:
        self.path = scheduler_file
        self.on_intent = on_intent
        self.is_channel_busy = is_channel_busy or (lambda _cid: False)
        self.tick_seconds = tick_seconds
        try:
            self._greet_p = min(max(float(greet_probability), 0.0), 1.0)  # chance a plain come-on is greeted
        except (TypeError, ValueError):
            self._greet_p = 1.0
        self.schedules: list[dict] = []
        self._prev: dict[str, dict] = {}
        self._primed = False  # first on_presence tick records baseline only (no firing); see on_presence
        self._user_fired: dict[tuple[str, str], datetime.datetime] = {}  # (schedule_id, user_key) -> last greeting
        # Greeting flags MUST survive restarts: they were in-memory only, so every bot restart re-armed
        # greetings for everyone who was already online (we restarted ~6x in one afternoon and the channel
        # got re-greeted each time).
        self._greet_state_path = scheduler_file.with_name("greet_state.json")
        self._load_greet_state()
        self._mtime: float = 0.0
        self._load()

    # --- persistence ---
    def _load_greet_state(self) -> None:
        """Restore per-user greeting timestamps so a restart doesn't re-greet everyone who is online."""
        try:
            if not self._greet_state_path.exists():
                return
            raw = json.loads(self._greet_state_path.read_text(encoding="utf-8"))
            for combined, iso in (raw.get("user_fired") or {}).items():
                sid, _, key = combined.partition("\u0000")
                try:
                    self._user_fired[(sid, key)] = datetime.datetime.fromisoformat(iso)
                except ValueError:
                    continue
            logger.info("Loaded %d greeting flags", len(self._user_fired))
        except (OSError, ValueError) as exc:
            logger.error("greet state load failed: %s", exc)

    def _save_greet_state(self) -> None:
        """Write greeting flags, pruning anything older than a week so the file can't grow forever."""
        try:
            cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
            data = {f"{sid}\u0000{key}": ts.isoformat()
                    for (sid, key), ts in self._user_fired.items() if ts >= cutoff}
            tmp = self._greet_state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"user_fired": data}, indent=2), encoding="utf-8")
            tmp.replace(self._greet_state_path)
        except OSError as exc:
            logger.error("greet state save failed: %s", exc)

    def _load(self) -> None:
        try:
            if not self.path.exists():
                self.schedules = []
                return
            mtime = self.path.stat().st_mtime
            if mtime == self._mtime and self.schedules:
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.schedules = data.get("schedules", [])
            self._mtime = mtime
            logger.info("Loaded %d schedules", len(self.schedules))
        except (OSError, ValueError) as exc:
            logger.error("scheduler load failed (keeping %d in memory): %s", len(self.schedules), exc)

    def _save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"schedules": self.schedules}, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            self._mtime = self.path.stat().st_mtime
        except OSError as exc:
            logger.error("scheduler save failed: %s", exc)

    # --- main loops ---
    async def run_forever(self) -> None:
        while True:
            try:
                self._load()
                await self._tick_time_based()
            except Exception:  # noqa: BLE001
                logger.exception("scheduler tick error")
            await asyncio.sleep(self.tick_seconds)

    async def _tick_time_based(self) -> None:
        now = datetime.datetime.now()
        today = now.strftime("%Y-%m-%d")
        weekday = now.weekday()  # 0=Mon..6=Sun
        for sch in self.schedules:
            if not sch.get("enabled", True) or sch.get("type") != "time_based":
                continue
            days = [6 if d == 7 else d for d in sch.get("trigger_days", list(range(7)))]
            if weekday not in days:
                continue
            selected = self._select_time(sch, today)
            if selected is None:
                continue
            if sch.get("last_triggered") == today:
                continue
            if now.strftime("%H:%M") >= selected:
                await self._fire(sch)
                sch["last_triggered"] = today
                self._save()

    def _select_time(self, sch: dict, today: str) -> str | None:
        raw = sch.get("trigger_time", "")
        times = [t.strip() for t in raw.split(",") if t.strip()]
        if not times:
            return None
        if sch.get("_random_time_date") != today:
            sch["_selected_trigger_time"] = random.choice(times)
            sch["_random_time_date"] = today
        return self._norm(sch.get("_selected_trigger_time", times[0]))

    @staticmethod
    def _norm(hhmm: str) -> str:
        try:
            h, m = hhmm.split(":")
            return f"{int(h):02d}:{int(m):02d}"
        except (ValueError, AttributeError):
            return hhmm

    async def on_presence(self, records: list[dict]) -> None:
        """Called by the Discord client on presence updates with a per-user presence map. Detects
        transitions (coming online, and starting/switching activities) and fires matching schedules.

        Each record: {key, display, names(set[str] lowercased), status, platforms(list[str]),
        activities(list[{type, name}])}. Spotify track changes are NOT treated as new activities (only a
        listening *session* start counts), so per-user monitoring doesn't fire on every song.
        """
        cur = {r["key"]: r for r in records}
        prev = self._prev
        changes: dict[str, dict] = {}
        for key, rec in cur.items():
            p = prev.get(key)
            prev_acts = {(a["type"], a["name"]) for a in p["activities"]} if p else set()
            prev_types = {a["type"] for a in p["activities"]} if p else set()
            cur_acts = {(a["type"], a["name"]) for a in rec["activities"]}
            new_acts = {(t, n) for (t, n) in (cur_acts - prev_acts)
                        if t != "custom"                                          # ignore custom-status edits
                        and not (t == "listening" and "listening" in prev_types)}  # skip song-to-song churn
            changes[key] = {
                "came_online": p is None,                # absent last tick (we only hold online members)
                "new_acts": new_acts,                    # started/switched a real activity (not stops/retracks)
            }
        self._prev = cur
        if not self._primed:
            # First tick after startup records the baseline (who's already online/active) without firing.
            # Without this, every member online at boot reads as came_online and the monitor spams on restart.
            self._primed = True
            return
        if not any(c["came_online"] or c["new_acts"] for c in changes.values()):
            return
        now = datetime.datetime.now()
        for sch in self.schedules:
            if not sch.get("enabled", True) or sch.get("type") != "condition_based":
                continue
            pcfg = self._presence_cfg(sch)
            matches = self._matching_users(pcfg, cur, changes)
            if not matches:
                continue
            if self.is_channel_busy(str(sch.get("channel_id", ""))):
                continue  # yield to a live conversation; try again next time
            if pcfg["per_user_cooldown"]:
                sid = str(sch.get("id", ""))
                # ONE greeting per user per cooldown window (24h by default), no matter what triggered it.
                # Previously a real activity start/switch fired unconditionally and never flagged the user,
                # so anyone who launched or switched games got greeted again and again (one member twice in 10
                # minutes, another twice in an hour). Activity changes still outrank plain come-ons for
                # WHICH user headlines the post -- they're more interesting -- but they no longer bypass the
                # window. A plain come-on additionally rolls the greeting probability; an activity start
                # doesn't (it's rarer and more worth saying something about).
                eligible = [(k, m) for (k, m) in matches if not self._user_in_cooldown(sch, k, now)]
                activity = [(k, m) for (k, m) in eligible if _is_activity_change(changes.get(k, {}))]
                greet = [(k, m) for (k, m) in eligible
                         if not _is_activity_change(changes.get(k, {}))
                         and changes.get(k, {}).get("came_online")
                         and random.random() < self._greet_p]
                fired_for = activity or greet
                if fired_for:
                    _headline_key, headline = fired_for[0]
                    await self._fire(sch, headline)
                    # Flag EVERY user this post covers (headline + anyone else it swept in) so none of them
                    # can be greeted again inside the window. Persisted, so a restart doesn't re-arm them.
                    for k, _m in fired_for:
                        self._user_fired[(sid, k)] = now
                    self._save_greet_state()
                # else: everyone matching is still inside their window, or the come-on roll lost -> silence
            else:
                if self._in_cooldown(sch, now):
                    continue
                if random.random() > sch.get("trigger_probability", 1.0):
                    continue
                await self._fire(sch, matches[0][1])
                sch["last_triggered"] = now.isoformat()
                self._save()

    @staticmethod
    def _presence_cfg(sch: dict) -> dict:
        """Normalize a schedule's presence conditions into a flat dict, supporting both the new `presence`
        hierarchy and the legacy `conditions.users_online` shape. Disabled filters resolve to None."""
        p = sch.get("presence")
        if p is None:
            users = {u.lower() for u in sch.get("conditions", {}).get("users_online", [])}
            return {"users": users, "new_activity": False, "activity_type": None,
                    "name_contains": None, "platform": None, "per_user_cooldown": False}
        filt = p.get("filters", {}) or {}

        def val(key):
            d = filt.get(key) or {}
            return d.get("value") if d.get("enabled") else None

        at, plat = val("activity_type"), val("platform")
        return {
            "users": {u.lower() for u in p.get("users_online", [])},
            "new_activity": bool(p.get("trigger_on_new_activity", False)),
            "activity_type": at.lower() if isinstance(at, str) else None,
            "name_contains": (val("activity_name_contains") or None),
            "platform": plat.lower() if isinstance(plat, str) else None,
            "per_user_cooldown": bool(p.get("per_user_cooldown", False)),
        }

    def _matching_users(self, pcfg: dict, cur: dict, changes: dict) -> list[tuple[str, dict]]:
        """All users whose presence change satisfies the schedule, as (user_key, match) pairs. `match`
        carries user/activity_name/activity_type/platform for prompt substitution.

        Trigger semantics:
          - new_activity + activity filters set -> ONLY a newly-started activity matching the filters
            fires it (e.g. fortnite_rally); coming online alone does NOT.
          - new_activity + no filters -> coming online or starting/switching a real activity fires it;
            custom-status edits and bare platform switches are ignored as noise.
          - no new_activity + filters set -> fires when the user comes online AND currently shows a
            matching activity.
          - no new_activity + no filters -> fires on coming online (plain greeting).
        """
        filtered = bool(pcfg["activity_type"] or pcfg["name_contains"])
        on_activity = pcfg["new_activity"]

        def _matches(t, n):
            return ((not pcfg["activity_type"] or t == pcfg["activity_type"])
                    and (not pcfg["name_contains"] or pcfg["name_contains"].lower() in (n or "").lower()))

        out: list[tuple[str, dict]] = []
        for key, rec in cur.items():
            ch = changes.get(key, {})
            if pcfg["users"] and not (rec["names"] & pcfg["users"]):
                continue
            if pcfg["platform"] and pcfg["platform"] not in {p.lower() for p in rec["platforms"]}:
                continue

            new_cand = sorted([(t, n) for (t, n) in ch.get("new_acts", set()) if _matches(t, n)])
            new_match = new_cand[0] if new_cand else None
            cur_match = next(((a["type"], a["name"]) for a in rec["activities"]
                              if _matches(a["type"], a["name"])), None)

            if on_activity and filtered:
                fired = new_match is not None
            elif on_activity:
                fired = bool(ch.get("came_online") or new_match)
            elif filtered:
                fired = bool(ch.get("came_online") and cur_match)
            else:
                fired = bool(ch.get("came_online"))
            if not fired:
                continue

            act = new_match or cur_match or (
                (rec["activities"][0]["type"], rec["activities"][0]["name"]) if rec["activities"] else None
            )
            out.append((key, {"user": self._display(rec),
                              "activity_type": act[0] if act else "",
                              "activity_name": act[1] if act else "",
                              "platform": pcfg["platform"] or (rec["platforms"][0] if rec["platforms"] else "")}))
        return out

    def _match_user(self, pcfg: dict, cur: dict, changes: dict) -> dict | None:
        """First matching user's match dict (or None). Thin wrapper over _matching_users."""
        m = self._matching_users(pcfg, cur, changes)
        return m[0][1] if m else None

    @staticmethod
    def _display(rec: dict) -> str:
        return rec.get("display") or next(iter(rec.get("names", set())), "")

    def _in_cooldown(self, sch: dict, now: datetime.datetime) -> bool:
        last = sch.get("last_triggered")
        if not last:
            return False
        try:
            last_dt = datetime.datetime.fromisoformat(last)
        except ValueError:
            return False
        hours = max(sch.get("cooldown_hours", 0), 1)  # min 1h for presence schedules
        return (now - last_dt) < datetime.timedelta(hours=hours)

    @staticmethod
    def _per_user_minutes(sch: dict) -> float:
        """Per-user cooldown window (minutes). Reads presence.per_user_cooldown_minutes, else falls back to
        cooldown_hours*60. 1-minute floor."""
        p = sch.get("presence") or {}
        m = p.get("per_user_cooldown_minutes")
        if m is None:
            m = float(sch.get("cooldown_hours", 1) or 1) * 60
        return max(float(m), 1.0)

    def _user_in_cooldown(self, sch: dict, user_key: str, now: datetime.datetime) -> bool:
        last = self._user_fired.get((str(sch.get("id", "")), user_key))
        if not last:
            return False
        return (now - last) < datetime.timedelta(minutes=self._per_user_minutes(sch))

    async def _fire(self, sch: dict, match: dict | None = None) -> None:
        logger.info("Firing schedule %s", sch.get("id", "?"))
        try:
            await self.on_intent(sch, match)
        except Exception:  # noqa: BLE001
            logger.exception("schedule %s intent failed", sch.get("id", "?"))
