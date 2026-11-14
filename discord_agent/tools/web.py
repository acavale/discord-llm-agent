"""web_search tool (DuckDuckGo via the ddgs package). Sync library, run in a thread."""

from __future__ import annotations

import asyncio
import logging
import re

from . import ToolContext, register

logger = logging.getLogger(__name__)


def _search_sync(query: str, max_results: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:  # package name fallback
        from duckduckgo_search import DDGS  # type: ignore
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _names_member(query: str, member_memory) -> str | None:
    """A known member's name or alias that appears in the query, or None."""
    users = (getattr(member_memory, "data", None) or {}).get("users", [])
    text = query.lower()
    for u in users:
        for n in [u.get("name", "")] + list(u.get("aliases", []) or []):
            n = str(n).strip().lower()
            if len(n) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", text):
                return n
    return None


@register(
    name="web_search",
    description="Search the web for current information. Use when you need facts, news, or anything you "
    "don't already know. Returns titles, snippets, and links.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    },
)
async def web_search(args: dict, ctx: ToolContext) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "(web_search: empty query)"
    if _names_member(query, ctx.member_memory):
        return ("(web_search refused: the query names someone in this server. Use member_memory for "
                "people here; their names don't go to a search engine.)")
    max_results = ctx.cfg.get("web_search", "max_results", default=8)
    try:
        results = await asyncio.to_thread(_search_sync, query, max_results)
    except Exception as exc:  # noqa: BLE001
        logger.error("web_search failed: %s", exc)
        return "(web_search failed)"
    if not results:
        return f"No results for '{query}'."
    out = [f"Results for '{query}':"]
    for r in results[:max_results]:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        out.append(f"- {title}: {body} ({href})")
    return "\n".join(out)
