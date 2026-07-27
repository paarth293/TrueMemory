"""TrueMemory Storage Layer
======================

Core data storage using SQLite with WAL mode. Manages the full database schema,
FTS5 full-text search triggers, and message CRUD operations.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default database location if none specified
DEFAULT_DB_PATH = Path.home() / ".truememory" / "memory.db"

# ---------------------------------------------------------------------------
# Standalone Helper Utilities
# ---------------------------------------------------------------------------

def _serialize_metadata(metadata: dict | None) -> str:
    """Serialize metadata dict into JSON string."""
    if metadata is None:
        return "{}"
    return json.dumps(metadata, sort_keys=True)


def _deserialize_metadata(raw: object) -> dict:
    """Parse JSON metadata string safely, defaulting to {} if invalid."""
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def select_message_cols(conn: sqlite3.Connection, alias: str = "") -> str:
    """Build standardized column list for SELECT queries."""
    prefix = f"{alias}." if alias else ""
    cols = ["id", "content", "sender", "recipient", "timestamp", "category", "modality", "directive", "metadata"]
    return ", ".join(f"{prefix}{c}" for c in cols)


def directive_filter_sql(conn: sqlite3.Connection, alias: str = "", include_directives: bool = False) -> str:
    """Return a SQL WHERE clause fragment to filter out standing directives during search."""
    if include_directives:
        return ""
    prefix = f"{alias}." if alias else ""
    return f" AND ({prefix}directive = 0 OR {prefix}directive IS NULL)"


# ---------------------------------------------------------------------------
# Schema DDL (Data Definition Language)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Core messages table: stores every captured memory and message
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    sender TEXT DEFAULT '',
    recipient TEXT DEFAULT '',
    timestamp TEXT DEFAULT '',
    category TEXT DEFAULT '',
    modality TEXT DEFAULT '',
    episode_id INTEGER DEFAULT NULL,
    emotional_valence REAL DEFAULT 0.0,
    embedding_separation BLOB DEFAULT NULL,
    directive INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

-- FTS5 virtual table for lightning-fast BM25 full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, sender, recipient, category, modality,
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Automated triggers to keep FTS5 index 100% in sync with messages table
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, sender, recipient, category, modality)
    VALUES (new.id, new.content, new.sender, new.recipient, new.category, new.modality);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content, sender, recipient, category, modality)
    VALUES (new.id, new.content, new.sender, new.recipient, new.category, new.modality);
END;

-- Entity profiles (L0 Personality Engram)
CREATE TABLE IF NOT EXISTS entity_profiles (
    entity TEXT PRIMARY KEY,
    message_count INTEGER DEFAULT 0,
    traits TEXT DEFAULT '{}',
    communication_style TEXT DEFAULT '{}',
    topics TEXT DEFAULT '[]',
    relationships TEXT DEFAULT '{}',
    updated_at TEXT
);

-- Fact timeline (L5 contradiction tracking and superseding facts)
CREATE TABLE IF NOT EXISTS fact_timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    fact TEXT NOT NULL,
    source_message_id INTEGER,
    timestamp TEXT,
    superseded_by INTEGER,
    entity_scope TEXT DEFAULT '',
    valid_from TEXT DEFAULT '',
    valid_to TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- Indexes for query optimization
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_directive ON messages(directive);
CREATE INDEX IF NOT EXISTS idx_fact_timeline_status ON fact_timeline(status);
"""


class Storage:
    """Manages SQLite database connection, table initialization, and message CRUD."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize storage manager and ensure schema DDL exists."""
        if db_path is None:
            db_path = os.environ.get("TRUEMEMORY_DB_PATH", DEFAULT_DB_PATH)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection configured with WAL mode and row dict factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Execute DDL statements to create database tables, indexes, and triggers."""
        with self.get_connection() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        log.info("Initialized TrueMemory database schema at %s", self.db_path)

    # -----------------------------------------------------------------------
    # CRUD Operations
    # -----------------------------------------------------------------------

    def add_message(
        self,
        content: str,
        sender: str = "",
        recipient: str = "",
        category: str = "",
        directive: bool = False,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Insert a new memory message into storage. Returns the new integer ID."""
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        metadata_json = _serialize_metadata(metadata)
        directive_int = 1 if directive else 0

        sql = """
        INSERT INTO messages (content, sender, recipient, category, directive, metadata, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """

        with self.get_connection() as conn:
            cursor = conn.execute(
                sql,
                (content, sender, recipient, category, directive_int, metadata_json, timestamp),
            )
            conn.commit()
            return cursor.lastrowid

    def get_message(self, message_id: int) -> dict[str, Any] | None:
        """Fetch a single message by ID."""
        sql = "SELECT * FROM messages WHERE id = ?;"
        with self.get_connection() as conn:
            row = conn.execute(sql, (message_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["metadata"] = _deserialize_metadata(result.get("metadata"))
            result["directive"] = bool(result.get("directive", 0))
            return result

    def delete_message(self, message_id: int) -> bool:
        """Delete a message by ID. Returns True if deleted, False if not found."""
        sql = "DELETE FROM messages WHERE id = ?;"
        with self.get_connection() as conn:
            cursor = conn.execute(sql, (message_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_directives(self) -> list[dict[str, Any]]:
        """Retrieve all standing directives (persistent system instructions)."""
        sql = "SELECT * FROM messages WHERE directive = 1 ORDER BY id ASC;"
        with self.get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["metadata"] = _deserialize_metadata(item.get("metadata"))
                item["directive"] = True
                results.append(item)
            return results

    def get_all_messages(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Retrieve all stored memory messages up to limit."""
        sql = "SELECT * FROM messages ORDER BY id DESC LIMIT ?;"
        with self.get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["metadata"] = _deserialize_metadata(item.get("metadata"))
                item["directive"] = bool(item.get("directive", 0))
                results.append(item)
            return results

    def count_messages(self) -> int:
        """Return total count of stored memory messages."""
        sql = "SELECT COUNT(*) FROM messages;"
        with self.get_connection() as conn:
            row = conn.execute(sql).fetchone()
            return row[0] if row else 0