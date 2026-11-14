"""recall tool: search the message archive (SQLite/FTS5) for past conversation."""

from __future__ import annotations

import asyncio
import datetime

from . import ToolContext, readable_channel_ids, register


@register(
    name="recall",
    description="Search past server messages by keyword to remember what was said before. Use when "
    "someone references an earlier conversation, a running joke, or something from another channel.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for."},
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
)
async def recall(args: dict, ctx: ToolContext) -> str:
    query = (args.get("query") or "").strip()
    limit = max(1, min(int(args.get("limit") or 8), 20))
    allowed = readable_channel_ids(ctx.guild)
    rows = await asyncio.to_thread(ctx.messages.search, query, limit, allowed)
    if not rows:
        return f"Nothing in the archive matches '{query}'."
    out = [f"Past messages matching '{query}':"]
    for r in rows:
        when = ""
        if r.get("ts"):
            when = datetime.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d")
        out.append(f"- [{when}] {r.get('username','?')}: {r.get('content','')}")
    return "\n".join(out)
