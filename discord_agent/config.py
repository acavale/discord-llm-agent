"""Configuration and secrets loading.

One config file (`config.yaml`) lives in the storage root: it holds no secrets and no machine-specific
paths. Per-machine values (secrets, the storage-root location) come from a local `.env` that is never
committed, so a dev instance (test token, test server) and a production instance can share one codebase
while keeping their credentials and data separate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Fallback storage roots, tried only when STORAGE_ROOT is not set in the environment.
_DRIVE_CANDIDATES = [
    Path.cwd() / "data",
]


def _detect_storage_root() -> Path | None:
    for candidate in _DRIVE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class Secrets:
    discord_token: str
    xai_api_key: str


@dataclass(frozen=True)
class Config:
    raw: dict
    storage_root: Path
    local_data_dir: Path
    secrets: Secrets

    def get(self, *keys, default=None):
        """Nested lookup: cfg.get('chime_in', 'enabled', default=False)."""
        cur = self.raw
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


def load_config(env_file: str | os.PathLike | None = None) -> Config:
    """Load .env, resolve the storage root, read config.yaml, and collect secrets.

    Raises RuntimeError with an actionable message on any missing piece.
    """
    load_dotenv(env_file)

    raw_root = os.environ.get("STORAGE_ROOT", "").strip()
    storage_root = Path(raw_root).expanduser() if raw_root else _detect_storage_root()
    if storage_root is None:
        raise RuntimeError(
            "STORAGE_ROOT is not set and ./data does not exist. "
            "Set STORAGE_ROOT in your .env (see .env.example)."
        )
    storage_root = storage_root.resolve()
    if not storage_root.exists():
        raise RuntimeError(f"STORAGE_ROOT does not exist: {storage_root}")

    raw_local = os.environ.get("LOCAL_DATA_DIR", "").strip()
    local_data_dir = (Path(raw_local).expanduser() if raw_local else Path.home() / ".discord_agent").resolve()
    local_data_dir.mkdir(parents=True, exist_ok=True)

    config_path = storage_root / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(
            f"config.yaml not found in storage root: {config_path}. "
            "Copy config.example.yaml there and fill it in."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    key = os.environ.get("XAI_API_KEY", "").strip()
    missing = [name for name, value in (("DISCORD_TOKEN", token), ("XAI_API_KEY", key)) if not value]
    if missing:
        raise RuntimeError(f"Missing secrets in environment/.env: {', '.join(missing)}")

    return Config(
        raw=raw,
        storage_root=storage_root,
        local_data_dir=local_data_dir,
        secrets=Secrets(discord_token=token, xai_api_key=key),
    )
