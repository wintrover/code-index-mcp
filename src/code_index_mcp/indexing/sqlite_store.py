"""
SQLite storage layer for deep code index data.

This module centralizes SQLite setup, schema management, and connection
pragmas so higher-level builders/managers can focus on data orchestration.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

SCHEMA_VERSION = 4


class SQLiteSchemaMismatchError(RuntimeError):
    """Raised when the on-disk schema cannot be used safely."""


class SQLiteIndexStore:
    """Utility wrapper around an on-disk SQLite database for the deep index."""

    def __init__(self, db_path: str) -> None:
        if not db_path or not isinstance(db_path, str):
            raise ValueError("db_path must be a non-empty string")
        self.db_path = db_path
        self._lock = threading.RLock()

    def initialize_schema(self) -> None:
        """Create database schema if needed and validate schema version."""
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with self.connect(for_build=True) as conn:
                self._create_tables(conn)
                self._ensure_schema_version(conn)
                # Ensure metadata contains the canonical project path placeholder
                if self.get_metadata(conn, "project_path") is None:
                    self.set_metadata(conn, "project_path", "")

    @contextmanager
    def connect(self, *, for_build: bool = False, timeout: float = 5.0) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager yielding a configured SQLite connection.

        Args:
            for_build: Apply write-optimized pragmas (journal mode, cache size).
            timeout: SQLite connection timeout in seconds.
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=timeout)
            conn.row_factory = sqlite3.Row
            self._apply_pragmas(conn, for_build)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def clear(self) -> None:
        """Remove existing database file."""
        with self._lock:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)

    # Metadata helpers -------------------------------------------------

    def set_metadata(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        """Persist a metadata key/value pair (value stored as JSON string)."""
        conn.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, json.dumps(value)),
        )

    def get_metadata(self, conn: sqlite3.Connection, key: str) -> Optional[Any]:
        """Retrieve a metadata value (deserialized from JSON)."""
        row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    # Internal helpers -------------------------------------------------

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                language TEXT,
                line_count INTEGER,
                imports TEXT,
                exports TEXT,
                package TEXT,
                docstring TEXT,
                content_hash TEXT,
                integrity_level TEXT
                    CHECK(integrity_level IN ('LOCAL_AST', 'GLOBAL_LINKED'))
                    DEFAULT 'GLOBAL_LINKED'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                symbol_id TEXT UNIQUE NOT NULL,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                type TEXT,
                line INTEGER,
                end_line INTEGER,
                signature TEXT,
                docstring TEXT,
                called_by TEXT,
                short_name TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbols_short_name ON symbols(short_name)
            """
        )

    def _ensure_schema_version(self, conn: sqlite3.Connection) -> None:
        stored = self.get_metadata(conn, "schema_version")
        if stored is None:
            self.set_metadata(conn, "schema_version", SCHEMA_VERSION)
            return

        version = int(stored)

        if version == 2:
            # v2 → v3: add content_hash and integrity_level columns
            conn.execute("ALTER TABLE files ADD COLUMN content_hash TEXT")
            conn.execute(
                "ALTER TABLE files ADD COLUMN integrity_level TEXT "
                "CHECK(integrity_level IN ('LOCAL_AST', 'GLOBAL_LINKED')) "
                "DEFAULT 'GLOBAL_LINKED'"
            )
            self.set_metadata(conn, "schema_version", 3)
            version = 3

        if version != SCHEMA_VERSION:
            raise SQLiteSchemaMismatchError(
                f"Unexpected schema version {version} (expected {SCHEMA_VERSION})"
            )

    def _apply_pragmas(self, conn: sqlite3.Connection, for_build: bool) -> None:
        pragmas: Dict[str, Any] = {
            "journal_mode": "WAL" if for_build else "WAL",
            "synchronous": "NORMAL" if for_build else "FULL",
            "cache_size": -262144,  # negative => size in KB, ~256MB
        }
        for pragma, value in pragmas.items():
            try:
                conn.execute(f"PRAGMA {pragma}={value}")
            except sqlite3.DatabaseError:
                # PRAGMA not supported or rejected; continue best-effort.
                continue
        if for_build:
            try:
                conn.execute("PRAGMA temp_store=MEMORY")
            except sqlite3.DatabaseError:
                pass

    # Public query helpers -------------------------------------------------

    def upsert_file_with_integrity(
        self,
        path: str,
        file_info: Any,
        symbols: list[Any],
        content_hash: str,
        integrity_level: str,
    ) -> None:
        """Insert or replace a file with integrity metadata and its symbols."""
        with self.connect() as conn:
            # Delete old symbols first – INSERT OR REPLACE creates a new file
            # row id, so the old symbols (referencing the old id) would cause
            # UNIQUE constraint violations on symbol_id if left in place.
            old_row = conn.execute(
                "SELECT id FROM files WHERE path = ?", (path,)
            ).fetchone()
            if old_row:
                conn.execute("DELETE FROM symbols WHERE file_id = ?", (old_row["id"],))
            # Upsert file
            conn.execute(
                """INSERT OR REPLACE INTO files
                   (path, language, line_count, imports, exports, package,
                    docstring, content_hash, integrity_level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    path,
                    file_info.language,
                    file_info.line_count,
                    json.dumps(file_info.imports) if file_info.imports else None,
                    json.dumps(file_info.exports) if file_info.exports else None,
                    file_info.package,
                    file_info.docstring,
                    content_hash,
                    integrity_level,
                ),
            )
            file_id = conn.execute(
                "SELECT id FROM files WHERE path = ?", (path,)
            ).fetchone()["id"]
            # Insert new symbols
            if symbols:
                conn.executemany(
                    """INSERT INTO symbols
                       (symbol_id, file_id, type, line, end_line, signature,
                        docstring, called_by, short_name)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            s.symbol_id,
                            file_id,
                            s.type,
                            s.line,
                            s.end_line,
                            s.signature,
                            s.docstring,
                            json.dumps(s.called_by) if s.called_by else None,
                            s.short_name,
                        )
                        for s in symbols
                    ],
                )

    def get_file_with_hash(
        self, path: str, content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get file row if it exists with matching content_hash."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE path = ? AND content_hash = ?",
                (path, content_hash),
            ).fetchone()
            return dict(row) if row else None

