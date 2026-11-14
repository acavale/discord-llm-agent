#!/usr/bin/env python3
"""Behavior check for the cross-channel + priority fixes.

Assembles the REAL production system prompt -- persona + member identity + quirks + the actual
general/game chatter pulled from the archive the same way discord_app._system_for does -- then probes
the three things we changed:

  1) DIRECT QUESTION: "you finally listening to game and general channels?"  -> must actually ANSWER,
     not dodge into observations. (This was a real failure seen live.)
  2) NO RECITAL: neutral chatter -> must NOT open with a status report about what people posted in
     general/game, no round-ups, no "also why is X posting..." asides.
  3) VARIETY: two separate presence-style greetings for different people -> must not reuse the same
     sentence skeleton.

Also prints prompt_tokens + time-to-first-token so the prompt-trim effect is measurable.

Usage: python scripts/behavior_check.py
No Discord, nothing is posted.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discord_agent.config import load_config
from discord_agent.context import build_system_prompt
from discord_agent.discord_app import phrase_cooldown_block, question_directive_block, recent_phrase_bans
from discord_agent.llm import LLM
from discord_agent.memory import MemberMemory, Quirks
from discord_agent.messages import Messages
from discord_agent.store import Store

cfg = load_config()
store = Store(cfg)
mm = MemberMemory(store.member_memory_file)
quirks = Quirks(store.quirks_file)
guild_id = str(cfg.get("bot", "guild_id", default=""))
msgs = Messages(store.guild_log(guild_id), store.index_db)
msgs.ingest_archive()

channels_cfg = cfg.get("channels", default={}) or {}
style_samples = int(cfg.get("context", "style_samples", default=0))
chatter = msgs.recent_chatter(set(channels_cfg), style_samples) if style_samples else None

persona = cfg.get("persona", "system_prompt", default="You are a Discord agent.")
model = cfg.get("llm", "ollama", "model", default="your-model")

llm = LLM(api_key="ollama", model=model, temperature=float(cfg.get("llm", "temperature", default=0.7)),
          timeout=int(cfg.get("llm", "ollama", "timeout", default=300)),
          base_url=cfg.get("llm", "ollama", "base_url", default="http://localhost:11434/v1"),
          extra_payload={
              "frequency_penalty": float(cfg.get("llm", "ollama", "frequency_penalty", default=0.0)),
              "presence_penalty": float(cfg.get("llm", "ollama", "presence_penalty", default=0.0)),
          })


banned = [str(p).lower().strip() for p in (cfg.get("context", "banned_phrases", default=None) or [])]


def _safe(s: str) -> str:
    """Windows consoles are cp1252; archived chat is full of emoji. Strip to ASCII for printing only."""
    return (s or "").encode("ascii", "replace").decode("ascii")


def system_for(present, prior_bot_msgs=None, question=None):
    """Mirrors discord_app.DiscordAgent._system_for: persona + identity + quirks + chatter, then the
    phrase-cooldown block and (when applicable) the direct-question directive."""
    system = build_system_prompt(persona, mm.format_identity(present), quirks.format_for_prompt(),
                                "agent-chat", "", chatter=chatter)
    bans = banned + [p for p in recent_phrase_bans(prior_bot_msgs or []) if p not in banned]
    system += phrase_cooldown_block(bans)
    if question:
        system += question_directive_block(question)
    return system


def ask(present, convo_text, prior_bot_msgs=None, question=None):
    system = system_for(present, prior_bot_msgs, question)
    t0 = time.perf_counter()
    r = llm.complete([{"role": "system", "content": system},
                      {"role": "user", "content": f"Recent conversation in this channel:\n{convo_text}"}])
    return (r.get("content") or "").strip(), time.perf_counter() - t0, system


print("=" * 76)
print(f"BEHAVIOR CHECK  model={model}")
print("=" * 76)
if chatter:
    print(f"\ngeneral/game lines injected as background ({len(chatter)}):")
    for u, c in chatter:
        print(f"  {_safe(u)}: {_safe(c)[:90]}")
else:
    print("\n!! no cross-channel chatter injected (style_samples=0) -- awareness test is meaningless")

# ---- 1) direct question ----
print("\n" + "-" * 76)
print("1) DIRECT QUESTION (must answer it, not dodge)")
print("-" * 76)
QUESTION = "you finally listening to game and general channels?"
reply, secs, system = ask({"alex"}, f"alex: {QUESTION}", question=QUESTION)
approx_prompt_tokens = int(len(system) / 4)
print(f"[prompt ~{approx_prompt_tokens} tokens | {secs:.1f}s]")
print(_safe(reply)[:600] or "(EMPTY)")
low = reply.lower()
answered = any(w in low[:160] for w in ("yeah", "yea", "yes", "kinda", "sort of", "i do", "i can",
                                        "always", "no ", "nope", "been"))
print(f"\n  -> opens with an actual answer? {'YES' if answered else 'NO -- still dodging'}")

# ---- 2) no recital ----
print("\n" + "-" * 76)
print("2) NO RECITAL (must not report what people posted elsewhere)")
print("-" * 76)
# Simulate a rut: prior bot messages leaning on the same phrases, so the cooldown list is non-empty.
PRIOR = ["the cube grim update is live and the vibe is real",
         "cube grim season is here so the vibe is peaking"]
reply2, secs2, sys2 = ask({"alex"}, "alex: im so bored rn", prior_bot_msgs=PRIOR)
print(f"[{secs2:.1f}s]  cooldown active: {'yes' if 'COOLDOWN' in sys2 else 'NO'}")
print(_safe(reply2)[:600] or "(EMPTY)")
recital_tells = [t for t in ("so the vibe", "just slid in", "also why is", "is officially",
                             "posting a", "just came online", "in jordan", "status")
                 if t in reply2.lower()]
print(f"\n  -> recital tells found: {recital_tells or 'none'}")
leaked = [b for b in ("cube grim", "the vibe") if b in reply2.lower()]
print(f"  -> banned phrases leaked: {leaked or 'none (good)'}")

# ---- 3) variety across greetings ----
print("\n" + "-" * 76)
print("3) VARIETY (two greetings must not share a skeleton)")
print("-" * 76)
g1, s1, _ = ask({"jordan"}, "jordan: (just came online, say something to them)", prior_bot_msgs=PRIOR)
g2, s2, _ = ask({"riley"}, "riley: (just came online, say something to them)",
                prior_bot_msgs=PRIOR + [g1])
print(f"[a {s1:.1f}s] {_safe(g1)[:250] or '(EMPTY)'}")
print(f"[b {s2:.1f}s] {_safe(g2)[:250] or '(EMPTY)'}")


def runs(s, n=5):
    w = re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


shared = runs(g1) & runs(g2)
print(f"\n  -> shared 5-word runs: {sorted(shared)[:3] if shared else 'none (good)'}")

print("\n" + "=" * 76)
print("DONE")
print("=" * 76)
