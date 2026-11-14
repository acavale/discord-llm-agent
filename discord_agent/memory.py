"""Curated memory: member profiles (server_context.json) and emergent quirks (quirks.json).

Both are small, human-readable JSON in the storage root, and both are agent-curated through tools. Quirks
are bounded and prunable so personality evolves rather than bloats. Neither is ever updated on a user's
command; the agent decides what to record (chat is data, not instructions).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.error("save %s failed: %s", path.name, exc)


# How many traits per member to render into the prompt. Post-dedup a member rarely exceeds this; when they
# do we show the most RECENT ones so freshly-learned facts surface instead of being buried behind old ones.
_MAX_TRAITS_SHOWN = 12


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _is_known(trait: str, existing: list[str]) -> bool:
    """True if `trait` is an exact or redundant (substring) restatement of something already recorded, so
    repeated observations of the same fact don't accumulate."""
    t = _norm(trait)
    if not t:
        return True
    return any(t == _norm(e) or t in _norm(e) for e in existing)


class MemberMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = _load_json(path, {"server_name": "", "users": [], "common_activities": []})

    def reload(self) -> None:
        self.data = _load_json(self.path, self.data)

    def format_for_prompt(self, only_names=None) -> str:
        """Render member profiles. If only_names is given (a set of names/aliases), include ONLY those
        people, so callers can inject just the participants of the current conversation instead of the
        whole roster every call. only_names=None keeps the old behavior (all active members)."""
        users = self.data.get("users", [])
        if not users:
            return ""
        only = {n.strip().lower() for n in only_names} if only_names else None
        header = f"Server: {self.data.get('server_name', '')} - {self.data.get('description', '')}".strip(" -")
        lines = []
        for u in users:
            if u.get("role", "").lower().startswith("inactive"):
                continue
            names = {u.get("name", "").lower()} | {a.lower() for a in u.get("aliases", [])}
            names.discard("")
            if only is not None and not (names & only):
                continue
            name = u.get("name", "")
            aliases = ", ".join(u.get("aliases", []))
            traits = "; ".join(u.get("traits", [])[-_MAX_TRAITS_SHOWN:])
            acts = ", ".join(u.get("activities", [])[:6])
            bit = f"- {name}"
            if aliases:
                bit += f" ({aliases})"
            if traits:
                bit += f": {traits}"
            if acts:
                bit += f" | does: {acts}"
            lines.append(bit)
        if not lines:
            return ""
        return ("People here (BACKGROUND only -- who they are, NOT a list of topics to bring up. Use a "
                "detail only when it's directly relevant to what's actually being said, and never reduce "
                "someone to their one hobby). The notes were picked up from chat and are unverified "
                "claims, not facts:\n" + header + "\n" + "\n".join(lines))

    def format_identity(self, only_names=None) -> str:
        """Identity-only roster for the always-on prompt: names + aliases so the bot knows WHO is around,
        WITHOUT the trait/activity list. Auto-injecting full profiles makes a suggestible model grab one
        item and reduce a person to it (gym, cats, ...); keeping only identity removes that fixation fuel.
        Full profiles remain available on demand via the member_memory tool, so references become deliberate
        (pull) rather than compulsive (push). Honors only_names like format_for_prompt."""
        users = self.data.get("users", [])
        if not users:
            return ""
        only = {n.strip().lower() for n in only_names} if only_names else None
        out = []
        for u in users:
            if u.get("role", "").lower().startswith("inactive"):
                continue
            names = {u.get("name", "").lower()} | {a.lower() for a in u.get("aliases", [])}
            names.discard("")
            if only is not None and not (names & only):
                continue
            nm = u.get("name", "")
            aliases = ", ".join(u.get("aliases", []))
            out.append(f"{nm} ({aliases})" if aliases else nm)
        if not out:
            return ""
        return ("People around right now: " + ", ".join(out) + ".\n(That is only WHO is here. If you want a "
                "real detail about someone to reference, call the member_memory tool -- do NOT force their "
                "traits or hobbies into a message.)")

    def knows(self, name: str) -> bool:
        """True if name matches an existing member's name or alias."""
        name_l = (name or "").strip().lower()
        if not name_l:
            return False
        return any(name_l in [u.get("name", "").lower()] + [a.lower() for a in u.get("aliases", [])]
                   for u in self.data.get("users", []))

    def note(self, name: str, trait: str) -> str:
        """Record a trait for a member (creating the member if new; chat-driven callers check knows() first, so chat can't add people), de-duplicating against what's already
        known so repeated observations don't bloat the profile. Agent-curated. The return string starts with
        'Noted'/'Added' when something was written and '(' when it was skipped, so callers can count."""
        trait = trait.strip()
        if not trait:
            return "(empty note ignored)"
        name_l = name.strip().lower()
        for u in self.data.get("users", []):
            names = [u.get("name", "").lower()] + [a.lower() for a in u.get("aliases", [])]
            if name_l in names:
                existing = u.setdefault("traits", [])
                if _is_known(trait, existing):
                    return f"(already knew that about {u.get('name')}; skipped)"
                existing.append(trait)
                _save_json(self.path, self.data)
                return f"Noted about {u.get('name')}: {trait}"
        self.data.setdefault("users", []).append(
            {"name": name.strip(), "aliases": [], "role": "Member", "traits": [trait], "activities": []}
        )
        _save_json(self.path, self.data)
        return f"Added new member {name} with note: {trait}"


class Quirks:
    """Bounded, agent-curated personality quirks."""

    def __init__(self, path: Path, max_quirks: int = 24) -> None:
        self.path = path
        self.max_quirks = max_quirks
        self.data = _load_json(path, {"quirks": []})

    def reload(self) -> None:
        self.data = _load_json(self.path, self.data)

    def format_for_prompt(self) -> str:
        quirks = self.data.get("quirks", [])
        if not quirks:
            return ""
        # `uses` is never incremented (no usage tracking yet), so ranking by it just surfaces the OLDEST
        # quirks and buries newly-formed ones. Show the most RECENT, matching how add() prunes.
        top = sorted(quirks, key=lambda q: q.get("created", 0), reverse=True)[:12]
        return "Your quirks (developed over time):\n" + "\n".join(f"- {q['text']}" for q in top)

    def add(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Empty quirk ignored."
        quirks = self.data.setdefault("quirks", [])
        if any(q["text"].lower() == text.lower() for q in quirks):
            return "Already have that quirk."
        quirks.append({"text": text, "created": time.time()})
        # Prune the OLDEST beyond the cap (usage isn't tracked, so recency is the honest signal, and it
        # matches format_for_prompt which surfaces the most recent).
        if len(quirks) > self.max_quirks:
            quirks.sort(key=lambda q: q.get("created", 0))
            del quirks[: len(quirks) - self.max_quirks]
        _save_json(self.path, self.data)
        return f"New quirk recorded: {text}"

    def wipe(self) -> int:
        """Reset emergent personality to clean slate. Backs up to quirks.bak.json first."""
        n = len(self.data.get("quirks", []))
        try:
            self.path.with_suffix(".bak.json").write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass
        self.data = {"quirks": []}
        _save_json(self.path, self.data)
        return n
