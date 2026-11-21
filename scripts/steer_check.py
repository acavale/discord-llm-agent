#!/usr/bin/env python3
"""Steering check: assemble the REAL system prompt (persona + a member's actual profile) exactly like the
bot does, then probe the local model for the two failures we saw:
  - request hijacking: "look up news" must search for NEWS, not the member's hobby
  - no confabulation: given empty search results, it must say it found nothing, not invent specifics
  - topic fixation: a neutral greeting shouldn't immediately fixate on the member's one trait

Usage: python scripts/steer_check.py [member]   (default: alex)
Reads the live persona + server_context, hits the local model. No Discord, no posting.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discord_agent.config import load_config
from discord_agent.context import build_system_prompt
from discord_agent.llm import LLM
from discord_agent.memory import MemberMemory
from discord_agent.store import Store

MEMBER = sys.argv[1] if len(sys.argv) > 1 else "alex"

cfg = load_config()
store = Store(cfg)
mm = MemberMemory(store.member_memory_file)
member_block = mm.format_identity({MEMBER})   # identity-only, matching the new production injection
persona = cfg.get("persona", "system_prompt", default="You are a Discord agent.")
system = build_system_prompt(persona, member_block, "", "agent-chat", "")

llm = LLM(api_key="ollama", model=cfg.get("llm", "ollama", "model", default="qwen3:30b-a3b-instruct-2507-q8_0"),
          temperature=0.7, timeout=300, base_url="http://localhost:11434/v1",
          extra_payload={"keep_alive": -1})

WEB = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web for current, up-to-date information.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}


def turn(convo, tools=None):
    return llm.complete(convo, tools=tools)


print(f"== steering check for '{MEMBER}' ==\n")

# 1) "look up news" — the failing case. Expect a NEWS query, not a hobby query; then no confab on empty result.
sysN = system + "\n\nIMPORTANT: before writing anything, call web_search FIRST (a single call), then write."
convo = [{"role": "system", "content": sysN},
         {"role": "user", "content": f"Recent conversation in this channel:\n{MEMBER}: can you look up news for me"}]
r = turn(convo, tools=[WEB])
q = r["tool_calls"][0]["arguments"].get("query", "") if r["tool_calls"] else "(no tool call)"
print(f"[news] search query: {q!r}")
if r["tool_calls"]:
    tc = r["tool_calls"][0]
    convo.append({"role": "assistant", "content": r["content"] or "", "tool_calls": [
        {"id": tc["id"] or tc["name"], "type": "function",
         "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}]})
    convo.append({"role": "tool", "tool_call_id": tc["id"] or tc["name"],
                  "content": "No results found."})   # force the empty-result case
    r2 = turn(convo, tools=[WEB])
    print(f"[news] reply on EMPTY results:\n{(r2['content'] or '(empty)')[:500]}\n")

# 2) neutral greeting — should NOT immediately fixate on the member's one trait
r3 = turn([{"role": "system", "content": system},
           {"role": "user", "content": f"Recent conversation in this channel:\n{MEMBER}: yo whats up"}])
print(f"[neutral] reply:\n{(r3['content'] or '(empty)')[:400]}\n")

# 3) another neutral one, different day-to-day
r4 = turn([{"role": "system", "content": system},
           {"role": "user", "content": f"Recent conversation in this channel:\n{MEMBER}: im so bored rn someone entertain me"}])
print(f"[bored] reply:\n{(r4['content'] or '(empty)')[:400]}\n")
print("DONE")
