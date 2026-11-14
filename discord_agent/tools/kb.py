"""KB notes: plain filesystem notes under the storage root's knowledge/ and research/ dirs.

Lightweight replacement for v1's RAG machinery: list, read, write, search. All paths are confined to the
KB root (no traversal). One config toggle (`kb_notes`) enables the whole group.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from . import ToolContext, register

_GROUP = "kb_notes"


_MAX_NOTE_CHARS = 20_000


def _kb_root(ctx: ToolContext) -> Path:
    return ctx.store.kb_dir


def _safe(root: Path, rel: str) -> Path | None:
    try:
        p = (root / rel).resolve()
        p.relative_to(root.resolve())
        return p
    except (ValueError, OSError):
        return None


@register(
    name="kb_list",
    description="List files in your knowledge base (notes/research you've saved).",
    parameters={"type": "object", "properties": {"subdir": {"type": "string"}}},
    group=_GROUP,
)
async def kb_list(args: dict, ctx: ToolContext) -> str:
    root = _kb_root(ctx)
    target = _safe(root, args.get("subdir", "") or "")
    if target is None or not target.exists():
        return "(empty or invalid path)"
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(names) if names else "(empty)"


@register(
    name="kb_read",
    description="Read a file from your knowledge base.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Relative path within the KB."}},
        "required": ["path"],
    },
    group=_GROUP,
)
async def kb_read(args: dict, ctx: ToolContext) -> str:
    p = _safe(_kb_root(ctx), args.get("path", ""))
    if p is None or not p.is_file():
        return "(file not found)"
    try:
        return await asyncio.to_thread(lambda: p.read_text(encoding="utf-8")[:10000])
    except OSError:
        return "(read error)"


@register(
    name="kb_write",
    description="Save or update a note/research file in your knowledge base (markdown).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path, e.g. research/topic.md"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    group=_GROUP,
)
async def kb_write(args: dict, ctx: ToolContext) -> str:
    p = _safe(_kb_root(ctx), args.get("path", ""))
    if p is None:
        return "(invalid path)"
    content = args.get("content", "") or ""
    if len(content) > _MAX_NOTE_CHARS:
        return f"(note too long: {len(content)} chars, max {_MAX_NOTE_CHARS})"

    def _write():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    try:
        await asyncio.to_thread(_write)
    except OSError:
        return "(write error)"
    return f"Saved {p.name} ({len(content)} chars)."


@register(
    name="kb_search",
    description="Search your knowledge base files for a phrase.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    group=_GROUP,
)
async def kb_search(args: dict, ctx: ToolContext) -> str:
    root = _kb_root(ctx)
    query = (args.get("query") or "").strip().lower()
    if not query:
        return "(empty query)"

    def _scan():
        hits = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in {".md", ".txt", ".json"}:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if query in text.lower() or query in p.name.lower():
                    hits.append(p.relative_to(root).as_posix())
        return hits

    hits = await asyncio.to_thread(_scan)
    return "Matches:\n" + "\n".join(f"- {h}" for h in hits) if hits else f"No KB files match '{query}'."
