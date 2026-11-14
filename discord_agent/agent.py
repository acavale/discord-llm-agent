"""The one agent loop. Messages in -> tool calls executed in a real loop -> final text or silence.

Holds a proper messages array. Tool results go back as tool-role messages so the model reasons across
tools in one thread. Silence is a first-class outcome in ambient mode (a stay_quiet tool, or empty text).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from . import tools as toolreg

logger = logging.getLogger(__name__)

_SILENCE_GUIDANCE = (
    "\n\nRight now you are deciding whether to speak in an ongoing conversation you were NOT addressed in. "
    "You mostly listen. Only speak if you genuinely have something worth adding: a real answer, a correction "
    "you're sure of, new information, or a line that actually lands. If not, call stay_quiet. Silence is the "
    "normal choice."
)

_STAY_QUIET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "stay_quiet",
        "description": "Say nothing. Use this whenever you don't have something genuinely worth adding.",
        "parameters": {"type": "object", "properties": {}},
    },
}

_SILENCE_TOKENS = {"stay_quiet", "stayquiet", "stay quiet", "[silent]", "(silent)", "silent"}


def _is_silence(text: str) -> bool:
    # The model sometimes writes the silence tool's NAME as plain text instead of calling it. Treat that
    # (and a few obvious markers) as silence so a bare tool name never leaks into the channel.
    norm = text.lower().strip(" .`'\"[]()*-\n")
    return norm in _SILENCE_TOKENS


# Hard cap on web_search calls per agent turn. The local model will otherwise chain many near-identical
# failing searches (e.g. hunting the "origin" of a nickname). Once spent, web_search is removed from the
# toolset for the rest of the turn so it must answer with what it has.
_SEARCH_CAP = 2


@dataclass
class AgentResult:
    text: str | None  # None = stayed silent / nothing to send


def _assistant_tool_msg(content: str, tool_calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": tc["id"] or tc["name"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
            }
            for tc in tool_calls
        ],
    }


async def run(
    llm,
    system_prompt: str,
    messages: list[dict],
    enabled_tools: set[str],
    ctx: "toolreg.ToolContext",
    allow_silence: bool = False,
    max_rounds: int = 6,
    force_first_tool: str | None = None,
) -> AgentResult:
    system = system_prompt + (_SILENCE_GUIDANCE if allow_silence else "")
    if force_first_tool:
        # The tool_choice forcing below is honored by Grok but IGNORED by Ollama's OpenAI endpoint, so also
        # state it as a hard instruction — the local model reliably follows a "call X first" directive.
        system += (f"\n\nIMPORTANT: before writing anything, call the `{force_first_tool}` tool FIRST (a "
                   f"single call), then use what it returns to write your reply. Do not answer without "
                   f"calling it first.")
    convo: list[dict] = [{"role": "system", "content": system}] + list(messages)

    schema_list = toolreg.schemas(enabled_tools)
    if allow_silence:
        schema_list = schema_list + [_STAY_QUIET_SCHEMA]

    searches = 0
    for _round in range(max_rounds):
        # Optionally force a specific tool on the FIRST round (e.g. make info/news posts web_search before
        # writing), then let the model decide for the rest.
        tool_choice = "auto"
        if _round == 0 and force_first_tool and schema_list and force_first_tool in enabled_tools:
            tool_choice = {"type": "function", "function": {"name": force_first_tool}}
        resp = await asyncio.to_thread(llm.complete, convo, schema_list or None, None, tool_choice)
        tool_calls = resp["tool_calls"]

        if not tool_calls:
            text = (resp["content"] or "").strip()
            if not text or _is_silence(text):
                return AgentResult(text=None)
            return AgentResult(text=text)

        convo.append(_assistant_tool_msg(resp["content"], tool_calls))
        for tc in tool_calls:
            if tc["name"] == "stay_quiet":
                return AgentResult(text=None)
            if tc["name"] == "web_search":
                if searches >= _SEARCH_CAP:
                    # The model can request several searches in one response; refuse the extras here
                    # rather than waiting for the schema trim at the end of the round.
                    convo.append({"role": "tool", "tool_call_id": tc["id"] or tc["name"],
                                  "content": "(search budget for this turn is used up; answer with what you have)"})
                    continue
                searches += 1
            result = await toolreg.dispatch(tc["name"], tc["arguments"], ctx)
            convo.append({"role": "tool", "tool_call_id": tc["id"] or tc["name"], "content": result})
        # Backstop against search spirals: once the per-turn search budget is spent, drop web_search from
        # the toolset so the model physically can't search again and must answer with what it has.
        if searches >= _SEARCH_CAP and schema_list:
            schema_list = [s for s in schema_list if s.get("function", {}).get("name") != "web_search"]

    # Out of tool rounds: force a final answer with no tools.
    resp = await asyncio.to_thread(llm.complete, convo, None)
    text = (resp["content"] or "").strip()
    if not text or _is_silence(text):
        return AgentResult(text=None)
    return AgentResult(text=text)


# ---------------------------------------------------------------------------
# Send-chain (agentic multi-message): instead of writing N messages in one shot and splitting on ---, the
# caller sends one message, then asks the model whether to send another, generating each follow-up as its own
# turn. This is that decision. Step 1 of the feature: defined and tested here, not yet wired into a send path.
# ---------------------------------------------------------------------------

_KEEP_GOING_SCHEMA = {
    "type": "function",
    "function": {
        "name": "keep_going",
        "description": (
            "Call this ONLY if, right now, you genuinely have a specific NEXT message worth sending on its "
            "own: a real follow-up thought, a reaction to what was just said, or a point you have not made "
            "yet. Do NOT call it to pad, to repeat yourself, or just to keep talking. If you are done, do "
            "not call it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "the specific thing your next message will add"},
            },
        },
    },
}

_FOLLOWUP_DECISION_PROMPT = (
    "You just sent the message(s) above. Decide whether to send ONE more message right now as a natural "
    "follow-up, or to stop here. Default to stopping: most of the time one message is enough and a real "
    "person would not keep firing off texts back to back. Only continue if there is a concrete reason in "
    "this exact moment (a distinct thought, a genuine reaction, something you actually left unsaid). If so, "
    "call keep_going; otherwise just reply with the single word DONE."
)


async def decide_followup(
    llm,
    system_prompt: str,
    messages: list[dict],
    sent_so_far: int,
    max_chain: int,
) -> bool:
    """Structured 'should I send another message?' decision for the send-chain. Returns True only when the
    model explicitly opts to continue (calls keep_going); defaults to False (stop) on the cap, on a plain
    DONE/answer, or on any error, so a chain always ends unless there is a deliberate reason to keep going."""
    if sent_so_far >= max(int(max_chain), 1):
        return False
    convo = (
        [{"role": "system", "content": system_prompt}]
        + list(messages)
        + [{"role": "user", "content": _FOLLOWUP_DECISION_PROMPT}]
    )
    try:
        resp = await asyncio.to_thread(llm.complete, convo, [_KEEP_GOING_SCHEMA], None, "auto")
    except Exception:  # noqa: BLE001
        logger.exception("followup decision failed; ending chain")
        return False
    return any(tc.get("name") == "keep_going" for tc in ((resp or {}).get("tool_calls") or []))
