"""Tool registry and dispatch. One registry; each tool is a module that registers a ToolSpec on import.

Adding a capability later = drop a module in this package, register it, and enable it in config. The agent
loop and the rest of the app never change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    cfg: object
    store: object
    messages: object        # discord_agent.messages.Messages
    member_memory: object   # discord_agent.memory.MemberMemory
    quirks: object          # discord_agent.memory.Quirks
    channel: object = None  # discord channel (for tools that post)
    guild: object = None
    llm: object = None      # discord_agent.llm.LLM (main agent model; may be local/Ollama)
    vision_llm: object = None  # vision-capable LLM for captioning (stays on Grok even when llm is local)


def readable_channel_ids(guild) -> set[str]:
    """Text channels visible to @everyone in this guild; empty when the guild can't be resolved.

    The bot reads with its own permissions, which are usually broader than a member's. Tools and the
    cross-channel context only look at these channels, so nobody can use the bot to read a channel they
    couldn't see themselves.
    """
    if guild is None:
        return set()
    everyone = guild.default_role
    out: set[str] = set()
    for ch in getattr(guild, "text_channels", []):
        try:
            if ch.permissions_for(everyone).read_messages:
                out.add(str(ch.id))
        except Exception:  # noqa: BLE001 -- unreadable permission state = not public
            continue
    return out


Handler = Callable[[dict, ToolContext], Awaitable[str]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Handler
    group: str = ""


REGISTRY: dict[str, ToolSpec] = {}


def register(name: str, description: str, parameters: dict, group: str | None = None):
    def deco(fn: Handler) -> Handler:
        REGISTRY[name] = ToolSpec(
            name=name, description=description, parameters=parameters, handler=fn, group=group or name
        )
        return fn

    return deco


def schemas(enabled: set[str]) -> list[dict]:
    out = []
    for name in enabled:
        spec = REGISTRY.get(name)
        if not spec:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
        )
    return out


async def dispatch(name: str, args: dict, ctx: ToolContext) -> str:
    spec = REGISTRY.get(name)
    if not spec:
        return f"(unknown tool: {name})"
    try:
        return await spec.handler(args or {}, ctx)
    except Exception as exc:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        return f"(tool {name} failed)"


def enabled_from_config(cfg) -> set[str]:
    tools_cfg = cfg.get("tools", default={}) or {}
    on_groups = {name for name, on in tools_cfg.items() if on}
    return {spec.name for spec in REGISTRY.values() if spec.group in on_groups or spec.name in on_groups}


# Import tool modules so they register. Keep at bottom to avoid circular imports.
from . import web, recall, memory_tools, kb, polls, vision  # noqa: E402,F401
