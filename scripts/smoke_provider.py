#!/usr/bin/env python3
"""Smoke-test a local Ollama model through the bot's OWN LLM client (discord_agent.llm.LLM), across several trials
because the tool-call bug is INTERMITTENT. Checks, per trial:
  - natural tool-call round-trip (decide -> call -> answer from result)
  - FORCED tool_choice (the force_search schedules force web_search on round 1)
  - tool SELECTION among several schemas (a poll request picks create_poll)
and on every generation asserts NO raw tool-call text (<tool_call>...) leaked into the content. With the
recovery parser in llm.py, leaks should be impossible and forced/natural calls should resolve to structured
calls even when the model emitted them as text.

Usage: python scripts/smoke_provider.py [model] [trials]   (defaults: qwen3:30b-a3b-instruct-2507-q8_0, 3)
Non-destructive: loads the model (keep_alive=-1) and generates short replies. No Discord, no posting.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discord_agent.llm import LLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:30b-a3b-instruct-2507-q8_0"
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

llm = LLM(api_key="ollama", model=MODEL, temperature=0.7, timeout=300,
          base_url="http://localhost:11434/v1", extra_payload={"keep_alive": -1})

WEB = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web for current, up-to-date information.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
POLL = {"type": "function", "function": {
    "name": "create_poll", "description": "Create a Discord poll with a question and a list of options.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}},
        "required": ["question", "options"]}}}
RECALL = {"type": "function", "function": {
    "name": "recall", "description": "Search the server's OWN past message archive for older context.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}


def leaked(text: str) -> bool:
    t = text or ""
    return "<tool_call>" in t or "</tool_call>" in t


def has(calls, name):
    return any(tc["name"] == name for tc in calls)


print(f"== model: {MODEL}  ({TRIALS} trials) ==")
fails = []

# banter voice (once) — should be plain text, no tool leak
r = llm.complete([
    {"role": "system", "content": "You are a Discord agent, a chill goofy Discord friend. short, lowercase, low-effort funny. no assistant tone."},
    {"role": "user", "content": "Recent conversation:\nnick: yo i finally beat that terraria boss after like 40 tries"},
])
print("\nbanter:", (r["content"] or "(EMPTY)")[:140])
if not (r["content"] or "").strip():
    fails.append("banter_empty")
if leaked(r["content"]):
    fails.append("banter_leak")

for t in range(TRIALS):
    # 1) natural tool-call
    msgs = [
        {"role": "system", "content": "You are a Discord agent with a web_search tool. For current events, call web_search FIRST, then answer."},
        {"role": "user", "content": "yo what's in the latest fortnite update"},
    ]
    a = llm.complete(msgs, tools=[WEB])
    if leaked(a["content"]):
        fails.append(f"natural_leak_t{t}")
    if has(a["tool_calls"], "web_search"):
        tc = a["tool_calls"][0]
        msgs.append({"role": "assistant", "content": a["content"] or "", "tool_calls": [
            {"id": tc["id"] or tc["name"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}]})
        msgs.append({"role": "tool", "tool_call_id": tc["id"] or tc["name"],
                     "content": "Fortnite v33.10: burst SMG buffs, new cube grim POI, drum shotgun vaulted."})
        b = llm.complete(msgs, tools=[WEB])
        if leaked(b["content"]):
            fails.append(f"natural_final_leak_t{t}")
        if not (b["content"] or "").strip():
            fails.append(f"natural_final_empty_t{t}")
    else:
        fails.append(f"natural_notool_t{t}")

    # 2) FORCED-via-directive: how the bot forces a search on Ollama (specific-function tool_choice is
    #    ignored here, so agent.py states it as a hard instruction). Must call web_search.
    f = llm.complete(
        [{"role": "system", "content": "You are a Discord agent writing a good-morning post. IMPORTANT: before writing anything, call the web_search tool FIRST (a single call), then use it to write. Do not answer without calling it first."},
         {"role": "user", "content": "post a quick good morning to the crew"}],
        tools=[WEB])
    if leaked(f["content"]):
        fails.append(f"forced_leak_t{t}")
    if not has(f["tool_calls"], "web_search"):
        fails.append(f"forced_notool_t{t}")

    # 3) tool SELECTION
    s = llm.complete(
        [{"role": "system", "content": "You are a Discord agent. Use the most appropriate tool for the request."},
         {"role": "user", "content": "make a poll asking who's down to game tonight — options: yes, nah, maybe"}],
        tools=[WEB, POLL, RECALL])
    if leaked(s["content"]):
        fails.append(f"select_leak_t{t}")
    if not has(s["tool_calls"], "create_poll"):
        fails.append(f"select_wrong_t{t}")

    print(f"trial {t}: natural={'ok' if f'natural_notool_t{t}' not in fails else 'FAIL'} "
          f"forced={'ok' if f'forced_notool_t{t}' not in fails else 'FAIL'} "
          f"select={'ok' if f'select_wrong_t{t}' not in fails else 'FAIL'}")

print("\n=== FAILS:", fails if fails else "NONE ===")
print("ALL OK" if not fails else "SOME FAILED")
sys.exit(0 if not fails else 1)
