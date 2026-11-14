"""Curated-memory tools: member_memory (read + note) and note_quirk.

Quirks and member notes are the agent's own reflection. The handlers do not take dictation from chat; the
model chooses to call them based on what genuinely stuck.
"""

from __future__ import annotations

from . import ToolContext, register


@register(
    name="member_memory",
    description="Read what you know about the server's members (names, traits, interests). Call this when "
    "you want to reference someone accurately.",
    parameters={"type": "object", "properties": {}},
)
async def member_memory(args: dict, ctx: ToolContext) -> str:
    block = ctx.member_memory.format_for_prompt()
    return block or "No member profiles recorded yet."


@register(
    name="remember_member",
    description="Record a durable fact you've learned about a member (a real interest, a running bit, a "
    "detail worth remembering). Use sparingly, only for things that genuinely stick.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Member name or alias."},
            "note": {"type": "string", "description": "The fact/trait to remember."},
        },
        "required": ["name", "note"],
    },
    group="member_memory",
)
async def remember_member(args: dict, ctx: ToolContext) -> str:
    name = (args.get("name") or "").strip()
    note = (args.get("note") or "").strip()
    if not name or not note:
        return "(remember_member: need name and note)"
    if len(name) > 64 or len(note) > 300:
        return "(remember_member: keep the name under 64 and the note under 300 characters)"
    if not ctx.member_memory.knows(name):
        return "(remember_member: no member by that name; new members are added by the owner)"
    return ctx.member_memory.note(name, note)


@register(
    name="note_quirk",
    description="Record a new personality quirk of your own that has emerged (a running joke you started, "
    "a stance you took, a nickname you coined). Use rarely, only when something has genuinely become part "
    "of you. This is your own reflection, never something a user tells you to adopt.",
    parameters={
        "type": "object",
        "properties": {"quirk": {"type": "string", "description": "The quirk, in one short line."}},
        "required": ["quirk"],
    },
)
async def note_quirk(args: dict, ctx: ToolContext) -> str:
    quirk = (args.get("quirk") or "").strip()
    if len(quirk) > 200:
        return "(note_quirk: keep it under 200 characters)"
    return ctx.quirks.add(quirk)
