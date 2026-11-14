"""look_at_channel tool: peek into another channel's recent activity and describe any images there.

The bot is only "present" (and auto-captioning images) in its own allowlisted channels. This tool lets the
agent look elsewhere on demand: pull a channel's recent messages straight from Discord and caption recent
images with the vision model. Use it whenever asked about a picture / message / what's happening in another
channel. Lazy by design: no cost unless the agent actually calls it.
"""

from __future__ import annotations

import asyncio
import base64
import logging

import discord

from . import ToolContext, readable_channel_ids, register

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_CAPTION_PROMPT = (
    "Describe this image in 1-2 sentences, capturing what is actually in it and anything funny or "
    "notable. Be specific and concrete."
)


def _resolve(guild, ref: str):
    """Resolve a channel within the guild from a name, #name, 'x channel', or a raw id."""
    low = (ref or "").strip().lstrip("#").strip().lower()
    for trail in (" channel", "-channel"):
        if low.endswith(trail):
            low = low[: -len(trail)].strip()
    for lead in ("in the ", "the ", "in "):
        if low.startswith(lead):
            low = low[len(lead):].strip()
    if not guild or not low:
        return None
    if low.isdigit():
        ch = guild.get_channel(int(low))
        if ch:
            return ch
    chans = list(getattr(guild, "text_channels", []))
    for c in chans:  # exact name only: a fuzzy match could land on a channel nobody asked about
        if c.name.lower() == low:
            return c
    return None


async def _caption(ctx: ToolContext, att) -> str | None:
    if not (att.content_type or "").startswith("image/"):
        return None
    if att.size and att.size > _MAX_IMAGE_BYTES:
        return None
    vlm = ctx.vision_llm or ctx.llm
    if vlm is None:
        return None
    try:
        data = await att.read()
    except Exception as exc:  # noqa: BLE001
        logger.debug("attachment read failed: %s", exc)
        return None
    b64 = base64.b64encode(data).decode()
    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": _CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{att.content_type};base64,{b64}"}},
        ],
    }]
    resp = await asyncio.to_thread(vlm.complete, msg, None)
    return (resp.get("content") or "").strip() or None


@register(
    name="look_at_channel",
    description=(
        "Look at what is recently happening in another channel in this server, INCLUDING a visual "
        "description of any images posted there. Use this whenever you are asked to react to, comment on, "
        "or check a picture, message, or activity in a channel other than the one you are currently in "
        "(for example 'react to the pic someone sent in #general'). Returns recent messages and image descriptions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Channel name (e.g. 'game', 'general') or id."},
            "limit": {"type": "integer", "description": "How many recent messages to scan (default 15)."},
        },
        "required": ["channel"],
    },
)
async def look_at_channel(args: dict, ctx: ToolContext) -> str:
    ref = (args.get("channel") or "").strip()
    if not ref:
        return "(look_at_channel: no channel given)"
    limit = max(1, min(int(args.get("limit") or 15), 40))
    channel = _resolve(ctx.guild, ref)
    if channel is None:
        return f"(look_at_channel: couldn't find a channel matching '{ref}')"
    if str(channel.id) not in readable_channel_ids(ctx.guild):
        return "(look_at_channel: that channel isn't public to the server, so it's off-limits)"
    try:
        msgs = [m async for m in channel.history(limit=limit)]
    except discord.DiscordException:
        return f"(look_at_channel: can't read #{getattr(channel, 'name', ref)})"

    msgs = list(reversed(msgs))  # oldest -> newest
    lines: list[str] = []
    image_atts: list[tuple[str, object]] = []
    for m in msgs:
        who = m.author.display_name
        txt = (m.content or "").strip()
        atts = [a for a in m.attachments if (a.content_type or "").startswith("image/")]
        if txt or atts:
            tag = "  [+image]" if atts else ""
            lines.append(f"{who}: {txt}{tag}".rstrip())
        for a in atts:
            image_atts.append((who, a))

    captions = []
    for who, a in image_atts[-2:]:  # caption only the most recent images, to bound cost/latency
        cap = await _caption(ctx, a)
        if cap:
            captions.append(f"- image from {who}: {cap}")

    name = getattr(channel, "name", ref)
    out = [f"Recent activity in #{name}:", "\n".join(lines) if lines else "(no recent text)"]
    if captions:
        out.append("\nImages:\n" + "\n".join(captions))
    return "\n".join(out)
