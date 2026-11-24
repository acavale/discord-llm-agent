"""Discord client: the glue. Routes messages to the one agent, runs the gate for ambient chime-in, feeds
presence to the scheduler, captions images, and handles the owner !inject command. Sending splits to the
2000-char limit. All response paths go through one _run_for_channel helper.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import re
import time
from collections import defaultdict

import discord

from . import agent as agentmod
from . import reflection as reflectmod
from . import tools as toolreg
from .context import Entry, Transcripts, build_system_prompt, build_user_messages
from .gate import Gate, RhythmBudget
from .llm import LLM
from .memory import MemberMemory, Quirks
from .messages import Messages
from .scheduler import Scheduler
from .tools import ToolContext

logger = logging.getLogger(__name__)

# Message separator for multi-message replies: a run of 3+ dashes (hyphen/en/em) standing alone, either on
# its own line or surrounded by whitespace inline. Lookarounds keep hyphenated words (e.g. "well-defined")
# and inline scores ("3---2") from splitting.
_MSG_SEP_RE = re.compile(r"(?<=\s)[-\u2013\u2014]{3,}(?=\s)|^[-\u2013\u2014]{3,}(?=\s|$)|(?<=\s)[-\u2013\u2014]{3,}$", re.MULTILINE)

_CONTROL_TOKEN_RE = re.compile(r"<\|.*?\|>")

# Words too common to be worth banning as part of a phrase (a rut is "the vibe", not "of the").
_PHRASE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "you", "your", "i", "im", "its", "it", "is", "was", "to", "of",
    "in", "on", "at", "for", "with", "that", "this", "just", "like", "so", "my", "me", "we", "they",
    "he", "she", "be", "are", "got", "get", "do", "not", "no", "if", "as", "about", "out", "up",
}

# A question aimed at the bot usually ends in '?' but the crew rarely punctuates, so also catch
# question-leading verbs/pronouns at the start of the line.
_QUESTION_LEAD_RE = re.compile(
    r"^(are|is|was|were|do|did|does|can|could|will|would|should|have|has|had|am|whats|what|who|when|"
    r"where|why|how|which|you)\b", re.IGNORECASE)


def recent_phrase_bans(bot_messages: list[str], limit: int = 14) -> list[str]:
    """2-3 word phrases that appear in 2+ of the given (recent) bot messages -- i.e. a rut forming.

    Module-level so the behavior-check script can build the exact same list the bot uses.
    """
    counts: dict[str, int] = {}
    for m in bot_messages:
        words = re.sub(r"[^a-z0-9 ]+", " ", (m or "").lower()).split()
        seen: set[str] = set()
        for n in (2, 3):
            for i in range(max(0, len(words) - n + 1)):
                chunk = words[i:i + n]
                if all(w in _PHRASE_STOPWORDS for w in chunk):
                    continue
                phrase = " ".join(chunk)
                if len(phrase) >= 7:
                    seen.add(phrase)
        for phrase in seen:
            counts[phrase] = counts.get(phrase, 0) + 1
    ranked = sorted((p for p, c in counts.items() if c >= 2), key=lambda p: (-counts[p], p))
    return ranked[:limit]


def greeting_skeleton(text: str, drop_tokens: set[str] | None = None) -> str:
    """A greeting's shape with the specifics stripped: normalized words minus the greeted person's
    name tokens, first 8. Two greetings sharing a skeleton are the same greeting in a different
    name, whatever the wording after the opening does."""
    words = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split()
    if drop_tokens:
        words = [w for w in words if w not in drop_tokens]
    return " ".join(words[:8])


def phrase_cooldown_block(bans: list[str]) -> str:
    if not bans:
        return ""
    return (
        "\n\nPHRASES ON COOLDOWN -- you have already used these recently (or they are permanently "
        "off-limits). Do NOT use any of them, in any form, in this message; say the thing a "
        "completely different way: " + ", ".join(f'"{b}"' for b in bans[:18])
    )


def question_directive_block(text: str) -> str:
    """Directive naming an actual question asked of the bot, or "" if the text isn't one."""
    text = (text or "").strip()
    if not text or len(text) > 400:
        return ""
    if "?" not in text and not _QUESTION_LEAD_RE.match(text):
        return ""
    return (
        f"\n\nTHE NEWEST MESSAGE IS A DIRECT QUESTION TO YOU: \"{text[:200]}\"\n"
        "Your FIRST sentence must actually answer it -- a real yes/no/the actual fact, in plain words. "
        "Only after answering may you joke or add anything. A joke instead of an answer, or a reply "
        "that leaves the question unaddressed, is a failed message."
    )


def _strip_control_tokens(text: str) -> str:
    """Drop model control/special tokens like <|eos|> that occasionally leak into generated content."""
    return _CONTROL_TOKEN_RE.sub("", text or "").strip()


# Appended to the system prompt on every turn AFTER the first in a send-chain, so a follow-up message
# continues the thought instead of repeating the line(s) the bot just sent.
_FOLLOWUP_TURN_HINT = (
    "\n\nThis is a FOLLOW-UP message in the same burst: you already sent your latest line(s), shown above as "
    "your most recent messages. Say only the next thing that genuinely adds something, keep it short, and do "
    "NOT repeat or rephrase what you just said. If someone replied in the meantime, react to that."
)

# Appended to every presence-greeting directive. The greeting RATE is set by the scheduler and is not
# touched here; this only varies the wording, because the greetings had collapsed onto one simile
# template reused on everyone.
_GREETING_VARIETY = (
    " Vary the SHAPE of the greeting itself: never open two greetings the same way, and never reuse a "
    "simile, metaphor, or sentence skeleton you have used on anyone else. Pick a different form each "
    "time -- a question, a callback to something they said or did, a short one-liner, a plain hello."
)


class DiscordAgent(discord.Client):
    def __init__(self, cfg, store, **kwargs) -> None:
        # Generated text must never ping @everyone/@here, roles, or arbitrary users. Replies still notify
        # the person being replied to.
        kwargs.setdefault("allowed_mentions", discord.AllowedMentions(
            everyone=False, roles=False, users=False, replied_user=True))
        super().__init__(**kwargs)
        self.cfg = cfg
        self.store = store
        self.prefix = cfg.get("bot", "command_prefix", default="!")
        self.creator_id = str(cfg.get("bot", "creator_user_id", default=""))
        self.guild_id = str(cfg.get("bot", "guild_id", default=""))
        self.persona_template = cfg.get("persona", "system_prompt", default="You are a Discord agent.")
        self.max_rounds = int(cfg.get("llm", "max_tool_rounds", default=6))
        self.max_len = int(cfg.get("context", "max_response_length", default=2000))
        self.transcript_n = int(cfg.get("context", "transcript_messages", default=30))
        self.style_samples = int(cfg.get("context", "style_samples", default=0))
        # Phrases permanently off-limits (model-invented catchphrases it anchors on). Editable in config.
        self.banned_phrases = [str(p).lower().strip() for p in
                               (cfg.get("context", "banned_phrases", default=None) or []) if str(p).strip()]
        # Skeletons of the greetings this bot has actually sent lately, persisted across restarts.
        # The variety instruction in the prompt did not stop the greetings collapsing onto one
        # opening frame, so _on_schedule enforces against this list the way _violates_ban enforces
        # the permanent bans: regenerate once, then drop the post.
        self._greet_skeleton_path = self.store.root / "greet_skeletons.json"
        self._greet_skeletons: list[str] = self._load_greet_skeletons()
        _w = cfg.get("context", "message_count_weights", default=None) or [0.5, 0.35, 0.15]
        _w = [float(x) for x in _w][:3]
        while len(_w) < 3:
            _w.append(0.0)
        self.msg_count_weights = _w if sum(_w) > 0 else [0.5, 0.35, 0.15]  # weights for 1/2/3-message replies
        self.chain_enabled = bool(cfg.get("context", "send_chain", default=False))  # step 3 switches paths to it
        self.chain_max = max(int(cfg.get("context", "send_chain_max", default=3)), 1)  # hard cap per send-chain
        # Post-chain reflection: after a real conversation, sometimes learn durable member facts from it.
        self.reflect_enabled = bool(cfg.get("context", "reflect_after_chat", default=True))
        self.reflect_prob = float(cfg.get("context", "reflect_probability", default=0.35))
        self.reflect_cooldown_s = float(cfg.get("context", "reflect_cooldown_seconds", default=600))
        self._last_reflect: dict[str, float] = {}  # cid -> monotonic ts of last reflection (per-channel throttle)

        self.llm_provider = str(cfg.get("llm", "provider", default="grok")).lower()
        _temp = float(cfg.get("llm", "temperature", default=0.7))
        if self.llm_provider == "ollama":
            self.llm = LLM(
                api_key="ollama",  # local Ollama ignores auth; any non-empty string works
                model=cfg.get("llm", "ollama", "model", default="qwen3:30b-a3b-instruct-2507-q8_0"),
                temperature=_temp,
                timeout=int(cfg.get("llm", "ollama", "timeout", default=120)),
                base_url=cfg.get("llm", "ollama", "base_url", default="http://localhost:11434/v1"),
                extra_payload={
                    "keep_alive": cfg.get("llm", "ollama", "keep_alive", default=-1),
                    "frequency_penalty": float(cfg.get("llm", "ollama", "frequency_penalty", default=0.0)),
                    "presence_penalty": float(cfg.get("llm", "ollama", "presence_penalty", default=0.0)),
                },
            )
        else:
            self.llm = LLM(
                api_key=cfg.secrets.xai_api_key,
                model=cfg.get("llm", "model", default="grok-4-fast-reasoning"),
                temperature=_temp,
            )
        # Image captioning always uses a vision-capable Grok model, even when the main model is local
        # (the local Instruct model is text-only). Cheap: only fires on images in active channels.
        self.vision_llm = LLM(
            api_key=cfg.secrets.xai_api_key,
            model=cfg.get("llm", "vision_model", default="grok-4-fast-reasoning"),
            temperature=_temp,
        )
        self.msgstore = Messages(store.guild_log(self.guild_id), store.index_db)
        self.member_memory = MemberMemory(store.member_memory_file)
        self.quirks = Quirks(store.quirks_file)
        self.transcripts = Transcripts(maxlen=self.transcript_n)
        self.gate = Gate(floor=float(cfg.get("chime_in", "gate_floor", default=0.5)))
        self.budget = RhythmBudget(
            hourly_budget=int(cfg.get("chime_in", "per_channel_hourly_budget", default=6)),
            post_speak_cooldown_s=float(cfg.get("chime_in", "post_speak_cooldown_seconds", default=30)),
        )
        self.name_aliases = [a.lower() for a in (cfg.get("chime_in", "name_aliases", default=None)
                             or ["agent", "bot"])]
        self.name_drop_prob = float(cfg.get("chime_in", "name_drop_probability", default=0.65))
        self.debounce_secs = float(cfg.get("chime_in", "debounce_seconds", default=2.5))
        self._pending: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._chain_cancel: dict[str, asyncio.Event] = {}  # cid -> set() asks an active send-chain to wrap up
        self.enabled_tools = toolreg.enabled_from_config(cfg)

        self.channels_cfg = cfg.get("channels", default={}) or {}
        self.channel_names = {cid: c.get("name", cid) for cid, c in self.channels_cfg.items()}

        self.scheduler = Scheduler(
            scheduler_file=store.scheduler_file,
            on_intent=self._on_schedule,
            is_channel_busy=lambda cid: self.transcripts.velocity(cid) >= 4,
            tick_seconds=int(cfg.get("scheduler", "tick_seconds", default=30)),
            greet_probability=float(cfg.get("scheduler", "greet_probability", default=1.0)),
        )
        self._scheduler_task: asyncio.Task | None = None

    # ---------- lifecycle ----------
    async def on_ready(self) -> None:
        logger.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "?")
        await asyncio.to_thread(self.msgstore.ingest_archive)
        if self.cfg.get("scheduler", "enabled", default=True) and self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self.scheduler.run_forever())
        await self._refresh_presence()
        logger.info("Tools enabled: %s", ", ".join(sorted(self.enabled_tools)))
        logger.info("LLM provider: %s (model %s)", self.llm_provider, self.llm.model)
        if self.llm_provider == "ollama":
            asyncio.create_task(self._warmup_llm())

    async def _warmup_llm(self) -> None:
        """Fire one tiny completion so Ollama loads the model into memory before the first real message;
        with keep_alive=-1 it then stays resident (no per-message reload). Best-effort."""
        try:
            await asyncio.to_thread(self.llm.complete, [{"role": "user", "content": "hi"}], None)
            logger.info("llm warm-up done: %s resident", self.llm.model)
        except Exception:  # noqa: BLE001
            logger.exception("llm warm-up failed (model will load on first message instead)")

    async def on_presence_update(self, before, after) -> None:
        await self._refresh_presence()

    async def _refresh_presence(self) -> None:
        guild = discord.utils.get(self.guilds, id=int(self.guild_id)) if self.guild_id else None
        if not guild:
            return
        groups = self._identity_groups()
        records = []
        for m in guild.members:
            if m.bot or str(m.status) == "offline":  # exclude bots: monitor humans only
                continue
            forms = {n.lower() for n in (m.name, m.display_name, getattr(m, "global_name", None),
                                         getattr(m, "nick", None)) if n}
            names = set(forms)
            # Alias bridge: if a member matches a known profile by any current name, treat ALL their
            # recorded aliases as present too, so a schedule keyed to an old handle still matches.
            for group in groups:
                if group & forms:
                    names |= group
            platforms = [p for p in ("desktop", "mobile", "web")
                         if str(getattr(m, f"{p}_status", "offline")) != "offline"]
            acts = [a for a in (self._activity_norm(act) for act in (m.activities or [])) if a]
            records.append({
                "key": str(m.id),
                "display": m.display_name,
                "names": names,
                "status": str(m.status),
                "platforms": platforms,
                "activities": acts,
            })
        self._presence_roster = self._presence_roster_text(records)
        await self.scheduler.on_presence(records)

    @staticmethod
    def _activity_norm(act) -> dict | None:
        """Normalize a discord Activity into {type, name}. type is the ActivityType name (playing,
        listening, streaming, watching, custom, competing). Returns None when there's no usable name."""
        try:
            atype = act.type.name.lower() if getattr(act, "type", None) is not None else "unknown"
        except AttributeError:
            atype = "unknown"
        if isinstance(act, discord.Spotify):
            name = f"{act.artist} - {act.title}"
        else:
            name = getattr(act, "name", "") or ""
        name = name.strip()
        return {"type": atype, "name": name} if name else None

    @staticmethod
    def _presence_roster_text(records: list[dict]) -> str:
        """Compact snapshot of who is online and what they are doing, grouped by activity, so a fired
        presence post can coordinate the whole room (squad people up, needle whoever's off alone)."""
        if not records:
            return ""
        by: dict[tuple, set] = {}
        idle: list[str] = []
        for r in records:
            acts = [a for a in (r.get("activities") or [])
                    if a["type"] in ("playing", "streaming", "watching", "competing", "listening")]
            if not acts:
                idle.append(r["display"])
                continue
            for a in acts:
                by.setdefault((a["type"], a["name"]), set()).add(r["display"])
        verb = {"playing": "playing", "listening": "listening to", "streaming": "streaming",
                "watching": "watching", "competing": "competing in"}
        lines = []
        for (atype, name), users in sorted(by.items(), key=lambda kv: (-len(kv[1]), min(kv[1]))):
            lines.append(f"- {verb.get(atype, atype)} {name}: {', '.join(sorted(users))}")
        if idle:
            lines.append(f"- online, nothing showing: {', '.join(sorted(idle))}")
        return "Right now in the server:\n" + "\n".join(lines)

    @staticmethod
    def _presence_event_line(match: dict) -> str:
        """The single line stating exactly what just changed. Used to anchor a presence-fired post on the
        real event (this person, this game/song) instead of a generic room summary."""
        user = match.get("user") or "someone"
        act = (match.get("activity_name") or "").strip()
        atype = (match.get("activity_type") or "").strip()
        if act and atype != "custom":
            verb = {"playing": "playing", "listening": "listening to", "streaming": "streaming",
                    "watching": "watching", "competing": "competing in"}.get(atype, "on")
            return (f"WHAT JUST CHANGED: {user} is now {verb} {act}. THIS is the event that triggered you, so "
                    f"open on it and talk about {user} and {act} specifically." + _GREETING_VARIETY)
        extra = f" (status: {act})" if act else ""
        return (f"WHAT JUST CHANGED: {user} just came online{extra}. THIS is the trigger, so say something "
                f"to {user} specifically -- riff on who they are, what they've been into lately, an inside "
                f"joke, the day/time, or whatever's actually going on. React to THEM, not the room, and MIX "
                f"IT UP: do NOT default to the 'you're online but not in a game, stop lurking, go boot "
                f"something' bit -- you have run that into the ground on everyone." + _GREETING_VARIETY)

    def _identity_groups(self) -> list[set[str]]:
        groups = []
        for u in self.member_memory.data.get("users", []):
            toks = {u.get("name", "").lower()} | {a.lower() for a in u.get("aliases", [])}
            toks.discard("")
            if toks:
                groups.append(toks)
        return groups

    # ---------- messages ----------
    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        if message.author.bot:  # ignore all other bots/webhooks: no replies, no archive, no triggers
            return
        is_dm = message.guild is None
        content = message.content or ""

        if content.startswith(self.prefix):
            await self._handle_command(message, is_dm)
            return

        cid = str(message.channel.id)
        clean = self._clean(message)
        # A human just spoke here: if a send-chain is mid-flight, tell it to wrap up so the bot doesn't
        # monologue over them. The normal handling below then responds to this new message.
        self._signal_chain_stop(cid)

        # DMs: answered for the owner and members of the bot's server; anyone outside it is ignored. Tools in
        # a DM still only see the server's public channels. No guild archive either way.
        if is_dm:
            if not await self._dm_allowed(message):
                return
            image_descs = await self._caption_images(message.attachments) if message.attachments else []
            self.transcripts.add(cid, Entry(user=message.author.display_name,
                                            content=clean or "[image]", image_descs=image_descs))
            self._cancel_pending(cid)
            await self._respond_now(message.channel, cid, allow_silence=False)
            return

        mentioned = self.user in message.mentions
        replied = bool(
            message.reference and getattr(message.reference, "resolved", None)
            and getattr(message.reference.resolved, "author", None) == self.user
        )
        chime = self.channels_cfg.get(cid, {}).get("chime_in", "off")
        chime_on = chime not in ("off", False, None)
        is_name_drop = self._is_name_drop(clean)

        # Caption images only in the allowlisted channels (where the bot actually responds). Everywhere else is
        # archive-only, so a caption would never be used -- skip the vision call.
        active = chime_on
        image_descs = await self._caption_images(message.attachments) if (active and message.attachments) else []

        # archive + transcript (every channel the bot can see, for recall + global awareness)
        if self.cfg.get("archive", "enabled", default=True):
            await asyncio.to_thread(
                self.msgstore.log, self.guild_id, cid, str(message.author.id),
                message.author.display_name, clean or "[image]", "text", False,
            )
        self.transcripts.add(cid, Entry(user=message.author.display_name, content=clean or "[image]",
                                        image_descs=image_descs))

        # PENNED to the configured allowlisted channels: everywhere else (general/game) the bot ONLY archives
        # + learns from the chatter, and NEVER responds -- not to @mentions, replies, or name-drops.
        if not chime_on:
            # READ-ONLY channels still drive LEARNING: reflect on this conversation so durable per-member
            # facts get recorded from the channels people actually talk in. Same throttle as elsewhere
            # (cooldown + probability + needs real conversation), and it never sends anything here.
            self._maybe_reflect(cid)
            return

        # Hard trigger: mention or reply. Answers immediately (in a allowlisted channel; cancel any pending
        # debounced reply first so we don't double up). The per-channel hourly budget is the only cap: a
        # direct address is answered unless the bucket is spent.
        if mentioned or replied:
            if not self.budget.has_budget(cid):
                logger.info("ch%s direct address skipped: hourly budget spent", cid)
                return
            self._cancel_pending(cid)
            await self._respond_now(message.channel, cid, allow_silence=False, consume_budget=True)
            return

        # Name-drop (typed name, no @): in a allowlisted channel only (general/game already returned above).
        # Also a direct address, so it is answered every time -- no probability roll; the hourly budget is
        # the only cap. Debounced like the rest. (Riddle answers land here too whenever they name the bot,
        # and as plain messages in a chime-all channel just below; both paths now always fire.)
        if is_name_drop:
            if not self.budget.has_budget(cid):
                logger.info("ch%s name-drop skipped: hourly budget spent", cid)
                return
            self._schedule_response(message.channel, cid, allow_silence=False, consume_budget=True)
            return

        # Ambient autonomy: only in chime channels.
        if not chime_on:
            return
        if chime == "all":  # respond to every message (forced; the persona decides how)
            self._schedule_response(message.channel, cid, allow_silence=False, consume_budget=False)
            return
        floor = 0.75 if chime == "limited" else None
        ok, reason = self.gate.should_consider(clean, self.transcripts, cid, floor=floor)
        if not ok or not self.budget.can_speak(cid):
            logger.debug("ch%s ambient skip: %s", cid, reason)
            return
        logger.info("ch%s ambient consider: %s", cid, reason)
        self._schedule_response(message.channel, cid, allow_silence=True, consume_budget=True)

    # ---------- the shared response path ----------
    async def _respond(self, channel, cid: str, allow_silence: bool, consume_budget: bool = False) -> None:
        if self.chain_enabled:
            system = self._system_for(cid, channel, self._present_users(cid))
            user_msgs = build_user_messages(self.transcripts.recent(cid, self.transcript_n))
            await self._emit_chain(channel, cid, system, user_msgs,
                                   allow_silence=allow_silence, consume_budget=consume_budget)
            self._maybe_reflect(cid)
            return
        system = self._system_for(cid, channel, self._present_users(cid))
        target = random.choices([1, 2, 3], weights=self.msg_count_weights, k=1)[0]
        system += self._msg_count_directive(target)
        user_msgs = build_user_messages(self.transcripts.recent(cid, self.transcript_n))
        ctx = ToolContext(self.cfg, self.store, self.msgstore, self.member_memory, self.quirks,
                           channel=channel, guild=self._tool_guild(channel), llm=self.llm, vision_llm=self.vision_llm)
        async with channel.typing():
            result = await agentmod.run(self.llm, system, user_msgs, self.enabled_tools, ctx,
                                        allow_silence=allow_silence, max_rounds=self.max_rounds)
            # Enforcement pass: if a permanently-banned catchphrase slipped through the prompt-level
            # cooldown, regenerate ONCE with the offender named. Costs one extra call, only when it fires.
            offender = self._violates_ban(result.text or "")
            if offender:
                logger.info("ch%s reply used banned phrase %r; regenerating once", cid, offender)
                retry_system = system + (
                    f"\n\nYour previous attempt used the forbidden phrase \"{offender}\". Write the message "
                    f"again from scratch WITHOUT that phrase or any variation of it, and without the same "
                    f"opening. Do not mention this instruction."
                )
                result = await agentmod.run(self.llm, retry_system, user_msgs, self.enabled_tools, ctx,
                                            allow_silence=allow_silence, max_rounds=self.max_rounds)
                if self._violates_ban(result.text or ""):
                    logger.info("ch%s retry still used a banned phrase; staying quiet", cid)
                    return
        if not result.text:
            return
        if self._too_repetitive(cid, result.text):
            logger.info("ch%s suppressed a near-duplicate reply (anti-loop)", cid)
            return
        if self._echoes_human(cid, result.text):
            logger.info("ch%s suppressed a reply that parroted a human line verbatim", cid)
            return
        clean = await self._send_multi(channel, result.text, cap=target)
        if consume_budget:
            self.budget.record_speak(cid)
        self.transcripts.add(cid, Entry(user="agent", content=clean, is_bot=True))
        if (getattr(channel, "guild", None) is not None
                and self.cfg.get("archive", "enabled", default=True)
                and self.cfg.get("archive", "log_bot_responses", default=True)):
            await asyncio.to_thread(self.msgstore.log, self.guild_id, cid, "agent", "agent",
                                    clean, "text", True)
        self._maybe_reflect(cid)   # live learning on the legacy path too, not just the send-chain path

    async def _emit_chain(self, channel, cid: str, system_first: str, messages_first: list[dict], *,
                          allow_silence: bool = False, consume_budget: bool = False,
                          force_first_tool: str | None = None, max_chain: int | None = None) -> str | None:
        """Send a response as a CHAIN of independent messages instead of one pre-split block. Each turn runs
        the agent (with its own tool calls), sends ONE message, records it, then asks decide_followup whether
        to send another, up to max_chain. After the first turn the system+messages are rebuilt from the live
        transcript, so the model sees what it just said and any reply that landed mid-chain is folded in. If a
        human speaks in the channel mid-chain, on_message sets the cancel event and the chain wraps up after
        its current message. Returns the last message sent, or None if nothing was sent."""
        cap = self.chain_max if max_chain is None else max(int(max_chain), 1)
        ctx = ToolContext(self.cfg, self.store, self.msgstore, self.member_memory, self.quirks,
                          channel=channel, guild=self._tool_guild(channel), llm=self.llm, vision_llm=self.vision_llm)
        system, messages = system_first, messages_first
        sent = 0
        last_clean: str | None = None
        cancel = asyncio.Event()
        self._chain_cancel[cid] = cancel
        try:
            while sent < cap:
                if cancel.is_set():  # a human spoke mid-chain: stop and hand back to the reactive path
                    break
                first = sent == 0
                async with channel.typing():
                    result = await agentmod.run(
                        self.llm, system, messages, self.enabled_tools, ctx,
                        allow_silence=(allow_silence if first else False),
                        max_rounds=self.max_rounds,
                        force_first_tool=(force_first_tool if first else None),
                    )
                clean = _strip_control_tokens(result.text or "")
                if not clean:
                    break  # silence on the first turn, or a follow-up came up empty -> end the chain
                if self._too_repetitive(cid, clean):
                    logger.info("ch%s chain: suppressed near-duplicate, ending chain (anti-loop)", cid)
                    break
                if not first:  # brief human-like beat before a follow-up lands
                    await asyncio.sleep(min(1.5, 0.5 + len(clean) / 220))
                await self._send(channel, clean)
                self.transcripts.add(cid, Entry(user="agent", content=clean, is_bot=True))
                if (getattr(channel, "guild", None) is not None
                        and self.cfg.get("archive", "enabled", default=True)
                        and self.cfg.get("archive", "log_bot_responses", default=True)):
                    await asyncio.to_thread(self.msgstore.log, self.guild_id, cid, "agent", "agent",
                                            clean, "text", True)
                if first and consume_budget:
                    self.budget.record_speak(cid)
                sent += 1
                last_clean = clean
                if sent >= cap:
                    break
                # Rebuild from the live transcript: folds in a reply that landed mid-chain and shows the model
                # its own just-sent line, so the next turn continues instead of repeating.
                present = self._present_users(cid)
                system = self._system_for(cid, channel, present or None) + _FOLLOWUP_TURN_HINT
                messages = build_user_messages(self.transcripts.recent(cid, self.transcript_n))
                if not await agentmod.decide_followup(self.llm, system, messages, sent, cap):
                    break
        finally:
            self._chain_cancel.pop(cid, None)
        return last_clean

    # ---------- post-chain reflection (durable member learning) ----------
    def _too_repetitive(self, cid: str, text: str) -> bool:
        """True if `text` closely repeats something the bot said in its last few messages here. Guards the
        local model's mode-collapse loop (posting the same line over and over, sometimes with just a number
        changed) AND its template loop (same sentence skeleton with the name swapped, e.g. "X just slid in
        like the vibe decided to ..." fired at three different people). Catches high token overlap or any
        shared 5-word run against the last 6 bot messages."""
        def _w(s):
            return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split()
        words = _w(text)
        if len(words) < 4:
            return False
        cur, cur_runs = set(words), {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}
        recent_bot = [e.content for e in self.transcripts.recent(cid, 12) if e.is_bot][-6:]
        for prev in recent_bot:
            pw = _w(prev)
            if len(pw) < 4:
                continue
            if len(cur & set(pw)) / max(1, len(cur | set(pw))) >= 0.5:
                return True
            if cur_runs & {" ".join(pw[i:i + 5]) for i in range(max(1, len(pw) - 4))}:
                return True
        return False

    def _recent_phrase_bans(self, cid: str, limit: int = 14) -> list[str]:
        """2-3 word phrases the bot has already leaned on in its recent messages here.

        Persona rules like 'don't anchor on a catchphrase' do not hold on this model -- it invented
        "cube grim" and "the vibe" and made them the spine of nearly every message. A concrete,
        per-turn list of phrases it just used is a narrow instruction, which this model follows far
        better than a general principle. Cheap: no extra LLM call.
        """
        msgs = [e.content for e in self.transcripts.recent(cid, 14) if e.is_bot][-6:]
        return recent_phrase_bans(msgs, limit)

    def _question_directive(self, cid: str) -> str:
        """If the newest human line is a question aimed at the bot, say so explicitly for THIS turn.

        The standing 'answer direct questions' persona rule gets outvoted by everything else in the
        prompt (it answered 'you finally listening to game and general channels?' with a phone joke and
        no answer). Naming the actual question in the system prompt is narrow enough to stick.
        """
        last_human = next((e for e in reversed(self.transcripts.recent(cid, 6)) if not e.is_bot), None)
        if not last_human:
            return ""
        return question_directive_block(last_human.content or "")

    def _violates_ban(self, text: str) -> str | None:
        """First permanently-banned phrase present in `text`, or None.

        The cooldown list is injected into the prompt, but this model ignores it often enough that
        "cube grim" and "slid in" still came out in testing. Prompt bans are prevention; this is
        enforcement -- the caller regenerates once, then gives up rather than posting a known tic.
        """
        low = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
        for phrase in self.banned_phrases:
            if phrase and phrase in low:
                return phrase
        return None

    def _load_greet_skeletons(self) -> list[str]:
        try:
            if not self._greet_skeleton_path.exists():
                return []
            raw = json.loads(self._greet_skeleton_path.read_text(encoding="utf-8"))
            return [str(s) for s in raw if str(s).strip()][-16:]
        except Exception as exc:
            logger.error("greet skeleton load failed: %s", exc)
            return []

    def _save_greet_skeletons(self) -> None:
        try:
            tmp = self._greet_skeleton_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._greet_skeletons[-16:]), encoding="utf-8")
            tmp.replace(self._greet_skeleton_path)
        except Exception as exc:
            logger.error("greet skeleton save failed: %s", exc)

    def _greet_drop_tokens(self, match: dict) -> set[str]:
        """Name tokens of the greeted user, plus their alias group, stripped before skeletonizing
        so the same frame aimed at different people still collides."""
        toks = set(re.sub(r"[^a-z0-9 ]+", " ", str(match.get("user") or "").lower()).split())
        for group in self._identity_groups():
            if toks & group:
                toks |= group
        return toks

    async def _enforce_greeting_variety(self, channel, sch: dict, instruction: str, text: str,
                                        match: dict, force_search: bool) -> str:
        """Enforce the greeting-variety rule the prompt alone could not hold: if this presence
        post's skeleton matches one of the last 16 greetings actually sent, regenerate once with
        the repeated frame named; if the retry repeats a skeleton too, drop the post. Silence
        beats a seventeenth copy of the same greeting. Returns the text to send, or "" to drop."""
        drop = self._greet_drop_tokens(match)
        skel = greeting_skeleton(text, drop)
        if len(skel.split()) >= 4 and skel in self._greet_skeletons:
            logger.info("schedule %s greeting repeated skeleton %r; regenerating once",
                        sch.get("id"), skel)
            retry = await self._run_for_channel(
                channel,
                instruction + (f"\n\nYour draft opened with a frame you have already used: "
                               f"\"{skel}...\". Open a completely different way -- different first "
                               f"words, different shape, no version of that opening."),
                force_search=force_search)
            if not retry or self._violates_ban(retry):
                logger.info("schedule %s greeting retry unusable; skipping the post", sch.get("id"))
                return ""
            skel2 = greeting_skeleton(retry, drop)
            if len(skel2.split()) >= 4 and skel2 in self._greet_skeletons:
                logger.info("schedule %s greeting retry repeated skeleton %r; skipping the post",
                            sch.get("id"), skel2)
                return ""
            text, skel = retry, skel2
        if len(skel.split()) >= 4:
            self._greet_skeletons.append(skel)
            self._save_greet_skeletons()
        return text

    def _echoes_human(self, cid: str, text: str) -> bool:
        """True if the reply parrots a recent human line back verbatim.

        Seen live: a member asked "you listening to me?" and the bot's post contained that exact line as its
        own sentence. The model treats the transcript as material to quote when it has nothing to add.
        Checks each line/sentence of the candidate against recent human messages.
        """
        def norm(s):
            return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()

        humans = {norm(e.content) for e in self.transcripts.recent(cid, 12) if not e.is_bot}
        humans = {h for h in humans if len(h.split()) >= 3}   # 1-2 word lines are coincidence, not echo
        if not humans:
            return False
        for piece in re.split(r"[\n.?!]+", text or ""):
            p = norm(piece)
            if len(p.split()) >= 3 and p in humans:
                return True
        return False

    def _maybe_reflect(self, cid: str) -> None:
        """After a conversational chain, sometimes reflect on it and durably record any new member facts.
        Throttled (per-channel cooldown + probability) and fired as a background task so it never adds
        latency to the reply. Skipped unless real humans were actually talking."""
        if not self.reflect_enabled:
            return
        now = time.monotonic()
        if now - self._last_reflect.get(cid, 0.0) < self.reflect_cooldown_s:
            return
        entries = self.transcripts.recent(cid, self.transcript_n)
        human_msgs = sum(1 for e in entries if not e.is_bot)
        if human_msgs < 3:  # not enough real conversation to learn anything durable
            return
        if random.random() >= self.reflect_prob:
            return
        self._last_reflect[cid] = now
        asyncio.create_task(self._reflect_after_chain(cid))

    async def _reflect_after_chain(self, cid: str) -> list[str]:
        """Reflection: extract durable facts from a channel's recent conversation and write them through the
        de-duplicating member memory (which updates the in-memory copy too, so later turns use them
        immediately). Returns the facts actually recorded. Best-effort; never raises into the caller."""
        try:
            entries = self.transcripts.recent(cid, self.transcript_n)
            present = {e.user for e in entries if not e.is_bot}
            if not present:
                return []
            evidence = "\n".join(
                f"{'agent' if e.is_bot else e.user}: {e.content}" for e in entries if e.content
            )
            known = self.member_memory.format_for_prompt(present)
            facts = await asyncio.to_thread(
                reflectmod.extract_member_facts, self.llm, evidence, known, sorted(present)
            )
            applied = reflectmod.apply_facts(self.member_memory, facts)
            if applied:
                logger.info("ch%s reflection recorded %d fact(s): %s", cid, len(applied), "; ".join(applied))
            return applied
        except Exception:  # noqa: BLE001
            logger.exception("post-chain reflection failed")
            return []

    # ---------- response scheduling (debounce + per-channel lock) ----------
    def _is_name_drop(self, text: str) -> bool:
        low = (text or "").lower()
        return any(alias in low for alias in self.name_aliases)

    async def _respond_now(self, channel, cid: str, allow_silence: bool, consume_budget: bool = False) -> None:
        # Per-channel lock serializes responses in a channel so two fast triggers can't overlap.
        async with self._locks[cid]:
            await self._respond(channel, cid, allow_silence=allow_silence, consume_budget=consume_budget)

    def _cancel_pending(self, cid: str) -> None:
        p = self._pending.pop(cid, None)
        if p and not p["task"].done():
            p["task"].cancel()

    def _signal_chain_stop(self, cid: str) -> None:
        """Ask an in-flight send-chain in this channel to stop after its current message (a human spoke)."""
        ev = self._chain_cancel.get(cid)
        if ev is not None:
            ev.set()

    def _schedule_response(self, channel, cid: str, allow_silence: bool, consume_budget: bool) -> None:
        # Debounce: coalesce a burst of fragmented messages into ONE reply after the channel goes quiet
        # for debounce_secs. Each new message cancels and restarts the wait, so it fires as soon as the
        # user pauses (not a fixed wait). A straggler that arrives after the reply already fired is just a
        # new trigger. Set debounce_seconds: 0 to disable.
        if self.debounce_secs <= 0:
            asyncio.create_task(self._respond_now(channel, cid, allow_silence, consume_budget))
            return
        prev = self._pending.get(cid)
        if prev and not prev["task"].done():
            prev["task"].cancel()
            allow_silence = allow_silence and prev["allow_silence"]     # a directed trigger (False) wins
            consume_budget = consume_budget and prev["consume_budget"]  # directed triggers don't spend budget
        task = asyncio.create_task(self._debounced_fire(channel, cid, allow_silence, consume_budget))
        self._pending[cid] = {"task": task, "allow_silence": allow_silence, "consume_budget": consume_budget}

    async def _debounced_fire(self, channel, cid: str, allow_silence: bool, consume_budget: bool) -> None:
        try:
            await asyncio.sleep(self.debounce_secs)
        except asyncio.CancelledError:
            return
        self._pending.pop(cid, None)
        recent = self.transcripts.recent(cid, 1)
        if recent and recent[0].is_bot:  # don't talk over itself
            return
        await self._respond_now(channel, cid, allow_silence, consume_budget)

    def _post_context(self, channel, instruction: str):
        """Shared setup for a standalone channel post (schedules + inject): the cid, the system prompt scoped
        to whoever's relevant (present speakers + anyone named in the instruction, else the full roster), and
        a neutral trigger turn. The instruction goes into the SYSTEM prompt as a PRIVATE directive with hard
        anti-echo rules -- a user turn gets quoted/reacted-to by suggestible models (it leaked the raw inject
        text into the channel), whereas a system directive is treated as the bot's own goal."""
        cid = str(channel.id)
        present = self._present_users(cid) | self._names_in_text(instruction)
        system = self._system_for(cid, channel, present or None)
        system += (
            "\n\nYou are composing ONE standalone message to post in this channel right now, unprompted, as a "
            "normal member. Below is a PRIVATE directive for what your message should do or say. It is NOT a "
            "chat message and NOT from anyone in the channel. NEVER quote it, restate it, paraphrase it, "
            "react to it, or acknowledge being asked/instructed/prompted -- output ONLY the finished message "
            "itself, in your own voice, as if the idea were your own.\n\nPRIVATE DIRECTIVE: " + instruction
        )
        return cid, system, [{"role": "user", "content": "Write your message now."}]

    async def _run_for_channel(self, channel, instruction: str, force_search: bool = False) -> str | None:
        """Single-shot post for a channel (schedules + inject): returns the text for the caller to send.
        force_search makes the model web_search before writing (info/news posts grounded in current facts)."""
        _cid, system, seed = self._post_context(channel, instruction)
        ctx = ToolContext(self.cfg, self.store, self.msgstore, self.member_memory, self.quirks,
                          channel=channel, guild=self._tool_guild(channel), llm=self.llm, vision_llm=self.vision_llm)
        result = await agentmod.run(self.llm, system, seed, self.enabled_tools, ctx,
                                    allow_silence=False, max_rounds=self.max_rounds,
                                    force_first_tool=("web_search" if force_search else None))
        return result.text

    async def _chain_for_channel(self, channel, instruction: str, force_search: bool = False) -> str | None:
        """Send-chain version of _run_for_channel (schedules/inject when send_chain is on): posts the seed as
        a chain that sends + records itself, serialized with conversational replies via the per-channel lock
        so the two can't interleave. Returns the last message sent (or None)."""
        cid, system, seed = self._post_context(channel, instruction)
        async with self._locks[cid]:
            return await self._emit_chain(channel, cid, system, seed, allow_silence=False,
                                          force_first_tool=("web_search" if force_search else None))

    def _system_for(self, cid: str, channel, present: set | None = None) -> str:
        name = self.channel_names.get(cid) or getattr(channel, "name", "channel")
        digest = self.transcripts.global_digest(cid, self.channel_names)
        public = self._public_channel_ids() if self.style_samples else set()
        chatter = (self.msgstore.recent_chatter(set(self.channels_cfg), self.style_samples,
                                                include_channel_ids=public)
                   if self.style_samples and public else None)
        system = build_system_prompt(
            self.persona_template,
            self.member_memory.format_identity(present),
            self.quirks.format_for_prompt(),
            name,
            digest,
            chatter=chatter,
        )
        bans = list(self.banned_phrases) + [p for p in self._recent_phrase_bans(cid)
                                            if p not in self.banned_phrases]
        system += phrase_cooldown_block(bans)
        system += self._question_directive(cid)
        return system

    def _present_users(self, cid: str) -> set:
        """Display names of the humans active in this channel's live transcript, for scoping which member
        profiles get injected (instead of dumping the whole roster every call)."""
        return {e.user for e in self.transcripts.recent(cid, self.transcript_n) if not e.is_bot}

    def _names_in_text(self, text: str) -> set:
        """Profiled names/aliases that literally appear in a string, so an inject or schedule that mentions
        a person pulls that person's profile in even if they haven't spoken."""
        low = (text or "").lower()
        found = set()
        for u in self.member_memory.data.get("users", []):
            for tok in [u.get("name", "")] + u.get("aliases", []):
                t = tok.lower().strip()
                if len(t) >= 3 and t in low:
                    found.add(t)
        return found

    # ---------- scheduler intent ----------
    @staticmethod
    def _schedule_wants_chain(sch: dict) -> bool:
        """Whether a scheduled post should use the multi-message send-chain. Off by default so long-form
        report schedules (and, conservatively, the presence monitor) stay single-shot; set \"chain\": true on
        a schedule to opt it in."""
        return bool(sch.get("chain", False))

    async def _on_schedule(self, sch: dict, match: dict | None = None) -> None:
        channel = self.get_channel(int(sch.get("channel_id", 0) or 0))
        if channel is None:
            logger.warning("schedule %s: channel not found", sch.get("id"))
            return
        instruction = sch.get("context") or sch.get("prompt") or sch.get("message") or ""
        if match:
            for k in ("user", "activity_name", "activity_type", "platform"):
                instruction = instruction.replace("{" + k + "}", str(match.get(k, "")))
            instruction = instruction.replace("{activity}", str(match.get("activity_name", "")))
            # Lead with the exact change that fired this so the post is about THAT, not a generic room
            # summary. The roster is demoted to clearly-labelled, do-not-recite background.
            event = self._presence_event_line(match)
            roster = getattr(self, "_presence_roster", "")
            roster_block = (
                "\n\nBackground only, who else is around right now (do NOT read this list out or turn it into a "
                "roll call; it's just so you can pull a same-game player in or react to who's actually "
                f"around):\n{roster}"
            ) if roster else ""
            instruction = f"{event}\n\n{instruction}{roster_block}"
            recent_bot = [e.content for e in self.transcripts.recent(str(channel.id), self.transcript_n)
                          if e.is_bot][-3:]
            if recent_bot:
                instruction += ("\n\nYou ALREADY posted these here recently. Do NOT repeat them or the same "
                                "kind of line/joke -- especially not another lurking / no-game roast. Take a "
                                "genuinely different angle:\n" + "\n".join(f"- {m}" for m in recent_bot))
        force_search = bool(sch.get("force_search", False))
        if self.chain_enabled and self._schedule_wants_chain(sch):
            await self._chain_for_channel(channel, instruction, force_search=force_search)
            return
        text = await self._run_for_channel(channel, instruction, force_search=force_search)
        offender = self._violates_ban(text or "")
        if offender:
            # Scheduled posts are unprompted, so silence is a perfectly fine outcome: retry once with the
            # offender named, then drop the post rather than fire off another catchphrase message.
            logger.info("schedule %s used banned phrase %r; regenerating once", sch.get("id"), offender)
            text = await self._run_for_channel(
                channel,
                instruction + (f"\n\nDo NOT use the phrase \"{offender}\" or any variation of it -- you have "
                               f"overused it. Take a completely different angle and opening."),
                force_search=force_search,
            )
            if self._violates_ban(text or ""):
                logger.info("schedule %s retry still used a banned phrase; skipping the post", sch.get("id"))
                return
        if not text:
            return
        if match:
            text = await self._enforce_greeting_variety(channel, sch, instruction, text, match,
                                                        force_search)
            if not text:
                return
        cid = str(channel.id)
        clean = await self._send_multi(channel, text)
        self.transcripts.add(cid, Entry(user="agent", content=clean, is_bot=True))
        if self.cfg.get("archive", "enabled", default=True):
            await asyncio.to_thread(self.msgstore.log, self.guild_id, cid, "agent", "agent", clean, "text", True)

    async def _dm_allowed(self, message) -> bool:
        """DMs are answered for the owner and for members of the bot's server; anyone else is ignored."""
        if self.creator_id and str(message.author.id) == self.creator_id:
            return True
        guild = self._server()
        if guild is None:
            return False
        if guild.get_member(message.author.id) is not None:
            return True
        # The member cache is not complete right after a restart, so a cache miss is not proof
        # someone is outside the server. Ask Discord once; a successful fetch also fills the cache.
        fetch = getattr(guild, "fetch_member", None)
        if fetch is None:
            return False
        try:
            return await fetch(message.author.id) is not None
        except discord.HTTPException:
            return False

    def _server(self):
        try:
            return self.get_guild(int(self.guild_id))
        except (TypeError, ValueError):
            return None

    def _tool_guild(self, channel):
        """The server tools should scope to. In a DM the channel has no guild, so use the bot's server."""
        return getattr(channel, "guild", None) or self._server()

    def _public_channel_ids(self) -> set[str]:
        """Channels visible to @everyone. Fails closed: if the server can't be resolved, nothing."""
        from .tools import readable_channel_ids
        try:
            guild = self.get_guild(int(self.guild_id))
        except (TypeError, ValueError):
            guild = None
        if guild is None:
            return set()
        return readable_channel_ids(guild) or set()

    # ---------- commands ----------
    async def _handle_command(self, message: discord.Message, is_dm: bool) -> None:
        body = message.content[len(self.prefix):].strip()
        if not body:
            return
        parts = body.split()
        cmd, args = parts[0].lower(), parts[1:]
        is_owner = str(message.author.id) == self.creator_id

        if cmd == "inject":
            await self._cmd_inject(message, args, is_dm, is_owner)
        elif cmd == "refresh" and is_owner:
            self.member_memory.reload()
            self.quirks.reload()
            await message.reply("reloaded my memory and quirks.")
        elif cmd in ("sch_list", "schlist") and is_owner:
            ids = [s.get("id", "?") for s in self.scheduler.schedules if s.get("enabled", True)]
            await message.reply("Active schedules:\n" + "\n".join(f"- {i}" for i in ids))
        elif cmd in ("sch_test", "schtest") and is_owner and args:
            sch = next((s for s in self.scheduler.schedules if s.get("id") == args[0]), None)
            if sch is None:
                await message.reply(f"no schedule called '{args[0]}'. check !sch_list.")
            else:
                await message.reply(f"triggering {args[0]}...")
                await self._on_schedule(sch)
        elif cmd == "wipe" and is_owner:
            await self._cmd_wipe(message, args)
        elif cmd == "reflect" and is_owner:
            await self._cmd_reflect(message)

    async def _cmd_wipe(self, message, args) -> None:
        """MIB-style wipe. Scopes: context (default), quirks, all. Never touches the archive or member
        profiles (those are canonical/curated; edit server_context.json directly if you really mean to)."""
        scope = (args[0].lower() if args else "context")
        done = []
        if scope in ("context", "all"):
            self.transcripts.clear()
            self.budget.reset()
            done.append("live context + rhythm budget")
        if scope in ("quirks", "all"):
            n = self.quirks.wipe()
            done.append(f"{n} quirks (backed up to quirks.bak.json)")
        if not done:
            await message.reply("usage: !wipe [context|quirks|all]  (member profiles are not wiped here)")
            return
        await message.reply("neuralyzed: " + "; ".join(done) + ". starting fresh.")

    async def _cmd_reflect(self, message) -> None:
        """Owner: force a memory-reflection pass on THIS channel's recent transcript right now (bypasses the
        probability/cooldown gate) and report what got recorded, to validate the learning loop on demand."""
        cid = str(message.channel.id)
        entries = self.transcripts.recent(cid, self.transcript_n)
        if sum(1 for e in entries if not e.is_bot) < 1:
            await message.reply("nothing in this channel's recent memory to reflect on yet.")
            return
        async with message.channel.typing():
            applied = await self._reflect_after_chain(cid)
        if applied:
            await message.reply("recorded:\n" + "\n".join(f"- {a}" for a in applied))
        else:
            await message.reply("reflected, nothing new worth keeping.")

    async def _cmd_inject(self, message, args, is_dm, is_owner) -> None:
        if not (is_dm and is_owner):
            return  # silent: inject is owner+DM only
        if not args:
            await message.reply("that's not how this works. !inject <#channel|id> [--test] [--silent] <what to say>")
            return
        target_ref = args[0]
        flags = {a for a in args if a.startswith("--")}
        rest = [a for a in args[1:] if not a.startswith("--")]
        prompt = " ".join(rest).strip()
        channel = self._resolve_channel(target_ref)
        if channel is None:
            await message.reply(f"`{target_ref}`? no idea what that is. give me the channel id or pick it "
                                f"from the # list so it links.")
            return
        if not prompt:
            await message.reply("inject what? you gave me a channel and no words.")
            return
        # --test (preview, never sent) and --silent (send but don't record) stay on the single-shot path;
        # a normal inject uses the send-chain when it's enabled.
        if "--test" in flags:
            text = await self._run_for_channel(channel, prompt)
            if not text:
                await message.reply("tried, came up empty. that prompt did me no favors.")
                return
            await message.reply(f"preview for #{getattr(channel, 'name', target_ref)} (not sent):\n"
                                f"{_strip_control_tokens(text)}")
            return
        if self.chain_enabled and "--silent" not in flags:
            last = await self._chain_for_channel(channel, prompt)
            if not last:
                await message.reply("tried, came up empty. that prompt did me no favors.")
                return
            await message.reply("done. it's live in there.")
            return
        text = await self._run_for_channel(channel, prompt)
        if not text:
            await message.reply("tried, came up empty. that prompt did me no favors.")
            return
        clean = await self._send_multi(channel, text)
        if "--silent" not in flags:
            self.transcripts.add(str(channel.id), Entry(user="agent", content=clean, is_bot=True))
            if self.cfg.get("archive", "enabled", default=True):
                await asyncio.to_thread(self.msgstore.log, self.guild_id, str(channel.id),
                                        "agent", "agent", clean, "text", True)
        await message.reply("done. it's live in there.")

    # ---------- helpers ----------
    def _split_into_messages(self, text: str, cap: int = 3) -> list[str]:
        """Split a conversational reply into up to `cap` messages on a --- separator: a run of 3+ dashes
        standing alone on its own line OR surrounded by whitespace inline (en/em dashes count too). Extra
        parts beyond the cap merge into the last. Used by chat, injects, and scheduled posts alike."""
        raw = (text or "").strip()
        parts = [p.strip() for p in _MSG_SEP_RE.split(raw)]
        parts = [p for p in parts if p]
        if not parts:
            return [raw]
        if len(parts) > cap:
            parts = parts[: cap - 1] + ["\n\n".join(parts[cap - 1:])]
        return parts

    @staticmethod
    def _msg_count_directive(n: int) -> str:
        """Per-reply instruction telling the model how many messages to send (count chosen upstream by the
        configured weights). Paired with a hard cap at the send layer so it can never exceed n."""
        if n <= 1:
            return ("\n\nFORMAT for this reply: send it as a SINGLE message. Do NOT use the --- separator; "
                    "one message only, even if it's short.")
        return (f"\n\nFORMAT for this reply: send EXACTLY {n} short messages, each separated by a line "
                f"containing only --- . Don't pad to hit the count; each message should earn its place.")

    def _resolve_channel(self, ref: str):
        """Resolve a channel from a <#id> mention, a raw id, or a #name / name (searches guilds)."""
        ref = (ref or "").strip()
        if ref.startswith("<#") and ref.endswith(">"):
            inner = ref[2:-1]
            return self.get_channel(int(inner)) if inner.isdigit() else None
        if ref.isdigit():
            return self.get_channel(int(ref))
        name = ref[1:] if ref.startswith("#") else ref
        for g in self.guilds:
            ch = discord.utils.get(g.text_channels, name=name)
            if ch:
                return ch
        return None

    def _clean(self, message: discord.Message) -> str:
        text = message.content or ""
        for m in message.mentions:
            text = text.replace(f"<@{m.id}>", "").replace(f"<@!{m.id}>", "")
        return text.strip()

    async def _caption_images(self, attachments) -> list[str]:
        descs: list[str] = []
        for att in (attachments or [])[:2]:
            if not (att.content_type or "").startswith("image/"):
                continue
            try:
                data = await att.read()
                b64 = base64.b64encode(data).decode()
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one short sentence for later reference."},
                        {"type": "image_url", "image_url": {"url": f"data:{att.content_type};base64,{b64}"}},
                    ],
                }]
                resp = await asyncio.to_thread(self.vision_llm.complete, msg, None)
                if resp.get("content"):
                    descs.append(resp["content"].strip())
            except Exception as exc:  # noqa: BLE001
                logger.debug("caption failed: %s", exc)
        return descs

    async def _send_multi(self, channel, text: str, cap: int = 3) -> str:
        """Send a reply/post as up to `cap` messages, splitting on the --- separator so the bot can fire off
        a couple of texts in a row (conversation, injects, and scheduled posts all use this). Discord has no
        '---' rule, so a bare dash run is always our separator. Returns the cleaned joined text for the
        transcript/archive."""
        text = _strip_control_tokens(text)
        parts = self._split_into_messages(text, cap)
        for i, part in enumerate(parts):
            if i:  # brief human-like gap + typing before each follow-up message
                async with channel.typing():
                    await asyncio.sleep(min(1.5, 0.5 + len(part) / 220))
            await self._send(channel, part)
        return "\n".join(parts)

    async def _send(self, channel, text: str) -> None:
        text = _strip_control_tokens(text)
        for chunk in self._chunks(text, self.max_len):
            try:
                await channel.send(chunk)
            except discord.DiscordException as exc:
                logger.error("send failed: %s", exc)
                break

    @staticmethod
    def _chunks(text: str, size: int):
        text = text.strip()
        while text:
            if len(text) <= size:
                yield text
                return
            cut = text.rfind("\n", 0, size)
            if cut <= 0:
                cut = size
            yield text[:cut]
            text = text[cut:].lstrip()
