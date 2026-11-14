"""Entry point: `python -m discord_agent` starts the bot. `python -m discord_agent --check` does an offline
config/storage sanity print without connecting to Discord.
"""

from __future__ import annotations

import logging
import sys

from .config import load_config
from .store import Store


def _check(cfg, store) -> int:
    print("discord_agent config OK")
    print(f"  storage_root : {cfg.storage_root}")
    print(f"  local_data   : {cfg.local_data_dir}")
    print(f"  config.yaml  : {store.config_file} ({'found' if store.config_file.exists() else 'MISSING'})")
    print(f"  scheduler    : {store.scheduler_file} ({'found' if store.scheduler_file.exists() else 'missing'})")
    print(f"  member memory: {store.member_memory_file} ({'found' if store.member_memory_file.exists() else 'missing'})")
    print(f"  model        : {cfg.get('llm', 'model', default='?')}")
    print(f"  guild        : {cfg.get('bot', 'guild_id', default='?')}")
    channels = cfg.get("channels", default={}) or {}
    print(f"  channels     : {', '.join(c.get('name', cid) for cid, c in channels.items()) or '(none)'}")
    from . import tools as toolreg

    print(f"  tools        : {', '.join(sorted(toolreg.enabled_from_config(cfg))) or '(none)'}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = load_config()
    except RuntimeError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 1

    store = Store(cfg)

    if "--check" in sys.argv[1:]:
        return _check(cfg, store)

    import discord

    from .discord_app import DiscordAgent

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    intents.presences = True

    client = DiscordAgent(cfg, store, intents=intents)
    client.run(cfg.secrets.discord_token, log_handler=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
