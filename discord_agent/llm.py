"""Thin xAI (Grok) client. OpenAI-compatible chat-completions with native tool calling and vision.

Synchronous on purpose; the async agent calls it via asyncio.to_thread. Returns a normalized dict so the
rest of the code never touches the raw API shape.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.x.ai/v1"  # xAI (Grok). Ollama: http://<host>:11434/v1 (OpenAI-compatible)

_TOOLCALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _recover_tool_calls_from_text(content: str, tool_names: frozenset = frozenset()) -> tuple[list[dict], str]:
    """Recover tool calls a model emitted as TEXT instead of via the structured tool_calls field, and strip
    them from the content so nothing tool-ish leaks into a channel. Local models (Qwen3 via Ollama) do this
    intermittently when the server's parser misses their native format. Handles Hermes-style
    <tool_call>{"name":...,"arguments":{...}}</tool_call> blocks and a whole-content bare JSON tool object.
    Returns (tool_calls, cleaned_content). Only called when the structured field came back empty."""
    if not content:
        return [], content

    def _norm(obj) -> dict | None:
        if not isinstance(obj, dict) or not obj.get("name"):
            return None
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        return {"id": "", "name": obj["name"], "arguments": args if isinstance(args, dict) else {}}

    calls: list[dict] = []
    if "<tool_call>" in content:
        for m in _TOOLCALL_TAG_RE.finditer(content):
            try:
                c = _norm(json.loads(m.group(1)))
            except ValueError:
                c = None
            if c:
                calls.append(c)
        cleaned = _TOOLCALL_TAG_RE.sub("", content).replace("<tool_call>", "").replace("</tool_call>", "").strip()
        if calls:
            return calls, cleaned
        content = cleaned  # tags were malformed/unparsable; at least don't leak the raw tags

    stripped = content.strip().strip("`").strip()
    if stripped.startswith("{") and '"name"' in stripped and ('"arguments"' in stripped or '"parameters"' in stripped):
        try:
            c = _norm(json.loads(stripped))
        except ValueError:
            c = None
        if c:
            return [c], ""
    # Last-ditch: a model narrating a call as text like  toolname("a", "b")  on its own line. Positional args
    # can't be reliably mapped back to params, so just STRIP the line so it never posts (the missed call is
    # backfilled by reflection/enrichment). Whole-line, known tool names only, to avoid false positives.
    if tool_names:
        pat = re.compile(r"(?m)^\s*(?:" + "|".join(re.escape(n) for n in tool_names) + r")\s*\([^\n]*\)\s*$")
        if pat.search(content):
            content = pat.sub("", content).strip()
    return [], content


class LLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.7, timeout: int = 60,
                 base_url: str = _DEFAULT_BASE_URL, extra_payload: dict | None = None,
                 max_tokens: int | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        # Optional per-response generation cap. On the local model this is the runaway guard: an uncapped
        # generation that falls into a repetition loop burns the whole request timeout and the reply dies
        # silently, while a capped one finishes (truncated) or fails fast. None = field omitted entirely.
        self.max_tokens = max_tokens
        self.url = base_url.rstrip("/") + "/chat/completions"
        # Provider-specific fields merged into every request payload (e.g. Ollama's keep_alive). Kept out
        # of the xAI path by default so we never send non-standard fields to Grok.
        self.extra_payload = dict(extra_payload or {})

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        tool_choice: str = "auto",
    ) -> dict:
        """Return {"content": str, "tool_calls": [{"id","name","arguments"}]}.

        `arguments` is a parsed dict. On error, returns content="" and no tool calls.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        payload.update(self.extra_payload)
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        try:
            resp = requests.post(
                self.url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") == "length":
                logger.warning("response hit the max_tokens cap (%s); output truncated", self.max_tokens)
        except Exception as exc:  # noqa: BLE001 - surface as empty, caller decides
            logger.error("xAI request failed: %s", exc)
            return {"content": "", "tool_calls": []}

        tool_calls = []
        for tc in message.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except (ValueError, KeyError):
                args = {}
            tool_calls.append({"id": tc.get("id", ""), "name": tc["function"]["name"], "arguments": args})

        content = message.get("content") or ""
        if "<think>" in content:
            # Reasoning models (e.g. Qwen3.8) emit <think>...</think> before the answer. Never show it in
            # chat: strip complete blocks and any stray tags (backstop alongside running in non-thinking mode).
            content = _THINK_RE.sub("", content)
            content = re.sub(r"</?think>", "", content).strip()
        # Fallback for local models that emit tool calls as TEXT (Qwen3 via Ollama) when the server parser
        # misses them: recover the call so the tool actually runs, and strip it so the raw call can't leak
        # into a channel. Only when the structured field is empty, so Grok's clean path is untouched.
        if not tool_calls:
            names = frozenset(t.get("function", {}).get("name", "") for t in (tools or [])) - {""}
            tool_calls, content = _recover_tool_calls_from_text(content, names)
        return {"content": content, "tool_calls": tool_calls}
