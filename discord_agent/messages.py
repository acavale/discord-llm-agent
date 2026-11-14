"""Message persistence: the canonical JSONL archive (shared, in Drive) plus a derived local SQLite/FTS5
index for fast recall and channel/user stats.

The JSONL is append-only and authoritative. The SQLite index is rebuildable from it and lives off Drive.
Every row carries channel_id (v1 logs lacked it; new rows have it). Archived content is data, never
instructions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class Messages:
    def __init__(self, jsonl_path: Path, index_db: Path) -> None:
        self.jsonl_path = jsonl_path
        self.db = sqlite3.connect(str(index_db), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                ts REAL,
                guild_id TEXT,
                channel_id TEXT,
                user_id TEXT,
                username TEXT,
                content TEXT,
                source TEXT,
                responded INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_msg_channel ON messages(channel_id, ts);
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(content, username, content='messages', content_rowid='rowid');
            """
        )
        self.db.commit()

    # --- ingest the existing JSONL archive once at startup (idempotent on id) ---
    def ingest_archive(self) -> int:
        if not self.jsonl_path.exists():
            return 0
        added = 0
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if self._insert(row):
                    added += 1
        self.db.commit()
        logger.info("Indexed %d archive messages from %s", added, self.jsonl_path.name)
        return added

    def _insert(self, row: dict) -> bool:
        ts = row.get("timestamp")
        ts_val = self._parse_ts(ts)
        try:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO messages (id, ts, guild_id, channel_id, user_id, username, content, source, responded) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    row.get("id"),
                    ts_val,
                    str(row.get("guild_id", "")),
                    str(row.get("channel_id", "")),
                    str(row.get("user_id", "")),
                    row.get("username", ""),
                    row.get("message", "") or "",
                    row.get("source", "text"),
                    1 if row.get("responded") else 0,
                ),
            )
            if cur.rowcount:
                self.db.execute(
                    "INSERT INTO messages_fts (rowid, content, username) "
                    "SELECT rowid, content, username FROM messages WHERE id = ?",
                    (row.get("id"),),
                )
                return True
        except sqlite3.Error as exc:
            logger.debug("index insert skipped: %s", exc)
        return False

    @staticmethod
    def _parse_ts(ts) -> float:
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            try:
                from datetime import datetime

                return datetime.fromisoformat(ts).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    # --- live logging ---
    def log(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        username: str,
        content: str,
        source: str = "text",
        responded: bool = False,
    ) -> None:
        ts = time.time()
        mid = f"{guild_id}_{channel_id}_{user_id}_{int(ts * 1000)}"
        row = {
            "id": mid,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "guild_id": str(guild_id),
            "channel_id": str(channel_id),
            "user_id": str(user_id),
            "username": username,
            "message": content,
            "source": source,
            "responded": responded,
        }
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("archive append failed: %s", exc)
        self._insert(row)
        self.db.commit()

    # --- recall ---
    def search(self, query: str, limit: int = 8, channel_ids=None) -> list[dict]:
        """Full-text search. channel_ids restricts results to those channels (an empty set returns
        nothing); None means no restriction."""
        q = query.strip()
        if not q:
            return []
        scope, params = "", []
        if channel_ids is not None:
            params = [str(c) for c in channel_ids]
            if not params:
                return []
            scope = f" AND m.channel_id IN ({','.join('?' * len(params))})"
        try:
            rows = self.db.execute(
                "SELECT m.username, m.content, m.ts, m.channel_id FROM messages_fts f "
                "JOIN messages m ON m.rowid = f.rowid "
                "WHERE messages_fts MATCH ?" + scope + " ORDER BY m.ts DESC LIMIT ?",
                (self._fts_query(q), *params, limit),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("fts search error: %s", exc)
            return []
        return [dict(r) for r in rows]

    @staticmethod
    def _fts_query(q: str) -> str:
        # Quote each term to avoid FTS5 syntax errors on punctuation.
        terms = [t for t in q.replace('"', " ").split() if t]
        return " OR ".join(f'"{t}"' for t in terms) or '""'

    def channel_stats(self, channel_id: str) -> dict:
        row = self.db.execute(
            "SELECT COUNT(*) n, MIN(ts) first, MAX(ts) last, COUNT(DISTINCT user_id) users "
            "FROM messages WHERE channel_id = ?",
            (str(channel_id),),
        ).fetchone()
        return dict(row) if row else {}

    def recent_chatter(self, exclude_channel_ids, limit: int = 15,
                       include_channel_ids=None) -> list[tuple[str, str]]:
        """Recent human messages from channels OTHER than the given ones (e.g. the crew's main chat), for
        use as a voice/style reference. Excludes the bot and pre-v2 rows that carry no channel_id."""
        exclude = [str(c) for c in (exclude_channel_ids or [])]
        ph = ",".join("?" * len(exclude)) if exclude else "''"
        inc_sql, inc = "", []
        if include_channel_ids is not None:
            inc = [str(c) for c in include_channel_ids]
            if not inc:
                return []
            inc_sql = f"AND channel_id IN ({','.join('?' * len(inc))}) "
        try:
            rows = self.db.execute(
                f"SELECT username, content FROM messages "
                f"WHERE channel_id != '' AND channel_id NOT IN ({ph}) "
                f"{inc_sql}"
                f"AND username != 'agent' AND length(trim(content)) > 1 "
                f"ORDER BY ts DESC LIMIT ?",
                (*exclude, *inc, int(limit)),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [(r["username"], r["content"]) for r in reversed(rows)]

    def close(self) -> None:
        self.db.close()
