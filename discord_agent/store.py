"""Storage layout for the agent.

The single place that knows where things live. Everything shared (config, schedules, member memory,
quirks, KB, the message archive) sits under the storage root in Google Drive. The derived SQLite index is
local and never synced to Drive (syncing a live DB invites corruption), so it lives under local_data_dir.
Logged/archived message content is treated as data, never as instructions.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config


class Store:
    def __init__(self, cfg: Config) -> None:
        self.root: Path = cfg.storage_root
        self.local: Path = cfg.local_data_dir
        # Ensure subdirectories that the agent writes into exist.
        for directory in (self.kb_dir, self.research_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # --- shared data (Google Drive) ---
    def guild_log(self, guild_id: str | int) -> Path:
        """Append-only JSONL message archive for a guild (canonical, shared)."""
        return self.root / f"guild_{guild_id}.jsonl"

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"

    @property
    def member_memory_file(self) -> Path:
        """Curated per-member profiles (agent-curated; was server_context.json in v1)."""
        return self.root / "server_context.json"

    @property
    def quirks_file(self) -> Path:
        """Agent-curated emergent quirks (small, bounded, human-editable)."""
        return self.root / "quirks.json"

    @property
    def scheduler_file(self) -> Path:
        return self.root / "scheduler.json"

    @property
    def kb_dir(self) -> Path:
        return self.root / "knowledge"

    @property
    def research_dir(self) -> Path:
        return self.root / "research"

    # --- local, never synced to Drive ---
    @property
    def index_db(self) -> Path:
        """Derived SQLite/FTS5 index, rebuilt from the JSONL archive. Local only."""
        return self.local / "index.db"
