"""create_poll tool: native Discord poll in the current channel."""

from __future__ import annotations

import datetime

from . import ToolContext, register

try:
    import discord
except ImportError:  # allows importing the module without discord installed (e.g. for tests)
    discord = None


@register(
    name="create_poll",
    description="Create a native Discord poll in the current channel. Use when a quick group vote fits the "
    "moment. Keep it to 2-6 short options.",
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "duration_hours": {"type": "integer", "description": "1-168 (default 24)."},
        },
        "required": ["question", "options"],
    },
)
async def create_poll(args: dict, ctx: ToolContext) -> str:
    if discord is None or ctx.channel is None:
        return "(create_poll unavailable here)"
    question = (args.get("question") or "").strip()
    options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()][:10]
    if not question or len(options) < 2:
        return "(create_poll needs a question and at least 2 options)"
    hours = max(1, min(int(args.get("duration_hours") or 24), 168))
    try:
        poll = discord.Poll(question=question, duration=datetime.timedelta(hours=hours))
        for opt in options:
            poll.add_answer(text=opt[:55])
        await ctx.channel.send(poll=poll)
    except Exception as exc:  # noqa: BLE001
        return "(poll failed)"
    return f"Posted a poll: {question}"
