"""
SQLite-backed index manager coordinating builder and store.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .sqlite_index_builder import SQLiteIndexBuilder
from .sqlite_store import SQLiteIndexStore, SQLiteSchemaMismatchError
from .strategies import StrategyFactory
from ..constants import INDEX_FILE_DB, INDEX_FILE, INDEX_FILE_SHALLOW, SETTINGS_DIR

logger = logging.getLogger(__name__)


class SQLiteIndexManager:
    """Manage lifecycle of SQLite-backed deep index."""

    def __init__(self) -> None:
        self.project_path: Optional[str] = None
        self.index_builder: Optional[SQLiteIndexBuilder] = None
        self.store: Optional[SQLiteIndexStore] = None
        self.temp_dir: Optional[str] = None
        self.index_path: Optional[str] = None
        self.shallow_index_path: Optional[str] = None
        self._shallow_file_list: Optional[List[str]] = None
        self._is_loaded = False
        self._last_build_stats: Dict[str, Any] = {}
        self._lock = threading.RLock()
        logger.info("Initialized SQLite Index Manager")

    def set_project_path(self, project_path: str, additional_excludes: Optional[List[str]] = None) -> bool:
        """Configure project path and underlying storage location.

        Args:
            project_path: Path to the project directory to index
            additional_excludes: Optional list of additional directory/file
                patterns to exclude from indexing (e.g., ['vendor', 'custom_deps'])

        Returns:
            True if configuration succeeded, False otherwise
        """
        with self._lock:
            if not project_path or not isinstance(project_path, str):
                logger.error("Invalid project path: %s", project_path)
                return False

            project_path = project_path.strip()
            if not project_path or not os.path.isdir(project_path):
                logger.error("Project path does not exist: %s", project_path)
                return False

            self.project_path = project_path
            project_hash = _hash_project_path(project_path)
            self.temp_dir = os.path.join(tempfile.gettempdir(), SETTINGS_DIR, project_hash)
            os.makedirs(self.temp_dir, exist_ok=True)

            self.index_path = os.path.join(self.temp_dir, INDEX_FILE_DB)
            legacy_path = os.path.join(self.temp_dir, INDEX_FILE)
            if os.path.exists(legacy_path):
                try:
                    os.remove(legacy_path)
                    logger.info("Removed legacy JSON index at %s", legacy_path)
                except OSError as exc:  # pragma: no cover - best effort
                    logger.warning("Failed to remove legacy index %s: %s", legacy_path, exc)

            self.shallow_index_path = os.path.join(self.temp_dir, INDEX_FILE_SHALLOW)
            self.store = SQLiteIndexStore(self.index_path)
            self.index_builder = SQLiteIndexBuilder(project_path, self.store, additional_excludes)
            self._is_loaded = False
            logger.info("SQLite index storage: %s", self.index_path)
            if additional_excludes:
                logger.info("Additional excludes: %s", additional_excludes)
            return True

    def build_index(
        self,
        force_rebuild: bool = False,
        max_workers: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        """Build or rebuild the SQLite index.

        Args:
            force_rebuild: Whether to force a full rebuild.
            max_workers: Maximum number of parallel workers.
                When None, defaults to min(4, cpu_count).
            timeout: Parallel build timeout in seconds.
                When None, scales dynamically with file count.
        """
        with self._lock:
            if not self.index_builder:
                logger.error("Index builder not initialized")
                return False
            try:
                stats = self.index_builder.build_index(
                    max_workers=max_workers,
                    timeout=timeout,
                )
            except SQLiteSchemaMismatchError:
                logger.warning("Schema mismatch detected; recreating database")
                self.store.clear()  # type: ignore[union-attr]
                stats = self.index_builder.build_index(
                    max_workers=max_workers,
                    timeout=timeout,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to build SQLite index: %s", exc)
                self._last_build_stats = {}
                self._is_loaded = False
                return False

            self._last_build_stats = dict(stats)

            logger.info(
                "SQLite index build complete: %s files, %s symbols",
                stats.get("files"),
                stats.get("symbols"),
            )

            if stats.get("timed_out"):
                logger.warning(
                    "Build timed out: %d of %d files processed",
                    stats.get("files", 0),
                    stats.get("total_files", 0),
                )
                self._is_loaded = False
                return False

            self._is_loaded = True
            return True

    def load_index(self) -> bool:
        """Validate that an index database exists and schema is current."""
        with self._lock:
            if not self.store:
                logger.error("Index store not initialized")
                return False
            try:
                self.store.initialize_schema()
                with self.store.connect() as conn:
                    metadata = self.store.get_metadata(conn, "index_metadata")
            except SQLiteSchemaMismatchError:
                logger.info("Schema mismatch on load; forcing rebuild on next build_index()")
                self._is_loaded = False
                return False
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to load SQLite index: %s", exc)
                self._is_loaded = False
                return False
            self._is_loaded = metadata is not None
            return self._is_loaded

    def refresh_index(
        self,
        max_workers: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        """Force rebuild of the SQLite index.

        Args:
            max_workers: Maximum number of parallel workers.
            timeout: Parallel build timeout in seconds.
        """
        with self._lock:
            logger.info("Refreshing SQLite deep index...")
            if self.build_index(
                force_rebuild=True,
                max_workers=max_workers,
                timeout=timeout,
            ):
                return self.load_index()
            return False

    def build_shallow_index(self) -> bool:
        """Build the shallow index file list using existing builder helper."""
        with self._lock:
            if not self.index_builder or not self.project_path or not self.shallow_index_path:
                logger.error("Index builder not initialized for shallow index")
                return False
            try:
                file_list = self.index_builder.build_shallow_file_list()
                with open(self.shallow_index_path, "w", encoding="utf-8") as handle:
                    json.dump(file_list, handle, ensure_ascii=False)
                self._shallow_file_list = file_list
                return True
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to build shallow index: %s", exc)
                return False

    def load_shallow_index(self) -> bool:
        """Load shallow index from disk."""
        with self._lock:
            if not self.shallow_index_path or not os.path.exists(self.shallow_index_path):
                return False
            try:
                with open(self.shallow_index_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list):
                    self._shallow_file_list = [_normalize_path(p) for p in data if isinstance(p, str)]
                    return True
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to load shallow index: %s", exc)
            return False

    def find_files(self, pattern: str = "*") -> List[str]:
        """Find files from the shallow index using glob semantics."""
        with self._lock:
            if not isinstance(pattern, str):
                logger.error("Pattern must be a string, got %s", type(pattern))
                return []
            pattern = pattern.strip() or "*"
            norm_pattern = pattern.replace("\\\\", "/").replace("\\", "/")
            regex = _compile_glob_regex(norm_pattern)

            if self._shallow_file_list is None:
                if not self.load_shallow_index():
                    if self.build_shallow_index():
                        self.load_shallow_index()

            files = list(self._shallow_file_list or [])
            if norm_pattern == "*":
                return files
            return [f for f in files if regex.match(f)]

    def get_file_summary(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Return summary information for a file.

        Tries the SQLite deep index first; falls back to lazy on-demand
        parse with content-hash based caching.
        """
        with self._lock:
            if not isinstance(file_path, str):
                logger.error("File path must be a string, got %s", type(file_path))
                return None

            # 1. Try existing SQLite index
            summary = self._get_from_sqlite(file_path)
            if summary:
                return summary

            # 2. Deep index not loaded or file not found → lazy parse
            return self._lazy_parse_and_cache(file_path)

    # ------------------------------------------------------------------
    # Lazy parse and cache helpers
    # ------------------------------------------------------------------

    def _read_file_safe(self, full_path: str) -> Optional[str]:
        """Read a file as UTF-8 text, returning None for binary or unreadable files."""
        try:
            with open(full_path, "rb") as raw:
                sample = raw.read(8192)
                if b"\x00" in sample:
                    return None
                raw.seek(0)
                with io.TextIOWrapper(raw, encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except OSError:
            return None

    def _lazy_parse_and_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse a single file on-demand. Content-hash based cache integrity."""
        if not self.project_path:
            return None

        full_path = os.path.join(self.project_path, file_path)
        if not os.path.isfile(full_path):
            return None

        content = self._read_file_safe(full_path)
        if content is None:
            return None

        current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 1. Cache hit: same hash + LOCAL_AST or higher → reuse
        cached = self._get_cached_with_hash(file_path, current_hash)
        if cached:
            return cached

        # 2. Cache miss: parse and store as LOCAL_AST
        ext = os.path.splitext(file_path)[1]
        strategy = StrategyFactory().get_strategy(ext)
        symbols_dict, file_info = strategy.parse_file(file_path, content)

        # Convert Dict[str, SymbolInfo] → list with symbol_id/short_name attrs
        # that upsert_file_with_integrity expects
        symbol_list = [
            SimpleNamespace(
                symbol_id=sym_id,
                type=s.type,
                line=s.line,
                end_line=s.end_line,
                signature=s.signature,
                docstring=s.docstring,
                called_by=s.called_by or [],
                short_name=sym_id.split("::", 1)[-1],
            )
            for sym_id, s in symbols_dict.items()
        ]

        self.store.upsert_file_with_integrity(
            path=file_path,
            file_info=file_info,
            symbols=symbol_list,
            content_hash=current_hash,
            integrity_level="LOCAL_AST",
        )

        return self._format_summary(
            file_path,
            file_info,
            symbols_dict,
            integrity_level="LOCAL_AST",
            content_hash=current_hash,
        )

    def _get_cached_with_hash(
        self, file_path: str, content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Return cache if hash matches and integrity_level is LOCAL_AST or higher.

        Returns None on hash mismatch or missing file.
        """
        if not self.store:
            return None

        row = self.store.get_file_with_hash(file_path, content_hash)
        if not row or row.get("integrity_level") not in ("LOCAL_AST", "GLOBAL_LINKED"):
            return None

        # Fetch associated symbols
        with self.store.connect() as conn:
            symbol_rows = conn.execute(
                """
                SELECT type, line, end_line, signature, docstring, called_by, short_name
                FROM symbols
                WHERE file_id = ?
                ORDER BY line ASC
                """,
                (row["id"],),
            ).fetchall()

        return self._format_from_row(row, symbol_rows)

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    def _format_summary(
        self,
        file_path: str,
        file_info: Any,
        symbols_dict: Dict[str, Any],
        integrity_level: str = "GLOBAL_LINKED",
        content_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format file summary with integrity metadata from parsed objects."""
        functions: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []

        for sym_id, s in symbols_dict.items():
            called_by = s.called_by or []
            short_name = sym_id.split("::", 1)[-1]
            info: Dict[str, Any] = {
                "name": short_name,
                "called_by": called_by,
                "line": s.line,
                "end_line": s.end_line,
                "signature": s.signature,
                "docstring": s.docstring,
            }
            sig = s.signature or ""
            if sig.startswith("def ") and "::" in sig:
                methods.append(info)
            elif sig.startswith("def "):
                functions.append(info)
            elif sig.startswith("class ") or s.type == "class":
                classes.append(info)
            else:
                if s.type == "method":
                    methods.append(info)
                elif s.type == "class":
                    classes.append(info)
                else:
                    functions.append(info)

        functions.sort(key=lambda item: item.get("line") or 0)
        classes.sort(key=lambda item: item.get("line") or 0)
        methods.sort(key=lambda item: item.get("line") or 0)

        language = file_info.language if hasattr(file_info, "language") else file_info.get("language")  # type: ignore[union-attr]
        line_count = file_info.line_count if hasattr(file_info, "line_count") else file_info.get("line_count")  # type: ignore[union-attr]
        imports = file_info.imports if hasattr(file_info, "imports") else file_info.get("imports", [])  # type: ignore[union-attr]
        exports = file_info.exports if hasattr(file_info, "exports") else file_info.get("exports", [])  # type: ignore[union-attr]
        docstring = file_info.docstring if hasattr(file_info, "docstring") else file_info.get("docstring")  # type: ignore[union-attr]

        return {
            "file_path": file_path,
            "language": language,
            "line_count": line_count,
            "symbol_count": len(symbols_dict),
            "functions": functions,
            "classes": classes,
            "methods": methods,
            "imports": imports or [],
            "exports": exports or [],
            "docstring": docstring,
            "meta": {
                "integrity_level": integrity_level,
                "is_global_indexed": integrity_level == "GLOBAL_LINKED",
                "content_hash": content_hash,
            },
            "capabilities": {
                "local_symbols": "AVAILABLE",
                "cross_references": (
                    "AVAILABLE"
                    if integrity_level == "GLOBAL_LINKED"
                    else "REQUIRES_DEEP_INDEX"
                ),
            },
        }

    def _format_from_row(
        self, row: Dict[str, Any], symbol_rows: Any
    ) -> Dict[str, Any]:
        """Format a cached SQLite row into the standard response structure."""
        imports = _safe_json_loads(row["imports"])
        exports = _safe_json_loads(row["exports"])
        categorized = _categorize_symbols(symbol_rows)
        integrity_level = row.get("integrity_level") or "GLOBAL_LINKED"

        return {
            "file_path": row["path"],
            "language": row["language"],
            "line_count": row["line_count"],
            "symbol_count": len(symbol_rows),
            "functions": categorized["functions"],
            "classes": categorized["classes"],
            "methods": categorized["methods"],
            "imports": imports,
            "exports": exports,
            "docstring": row["docstring"],
            "meta": {
                "integrity_level": integrity_level,
                "is_global_indexed": integrity_level == "GLOBAL_LINKED",
                "content_hash": row.get("content_hash"),
            },
            "capabilities": {
                "local_symbols": "AVAILABLE",
                "cross_references": (
                    "AVAILABLE"
                    if integrity_level == "GLOBAL_LINKED"
                    else "REQUIRES_DEEP_INDEX"
                ),
            },
        }

    # ------------------------------------------------------------------
    # Internal SQLite query
    # ------------------------------------------------------------------

    def _get_from_sqlite(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Query SQLite for file summary. Returns None if not ready or not found."""
        if not self.store or not self._is_loaded:
            if not self.load_index():
                return None

        normalized = _normalize_path(file_path)
        with self.store.connect() as conn:
            row = conn.execute(
                """
                SELECT id, language, line_count, imports, exports, docstring
                FROM files WHERE path = ?
                """,
                (normalized,),
            ).fetchone()

            if not row:
                return None

            symbol_rows = conn.execute(
                """
                SELECT type, line, end_line, signature, docstring, called_by, short_name
                FROM symbols
                WHERE file_id = ?
                ORDER BY line ASC
                """,
                (row["id"],),
            ).fetchall()

        imports = _safe_json_loads(row["imports"])
        exports = _safe_json_loads(row["exports"])
        categorized = _categorize_symbols(symbol_rows)

        return {
            "file_path": normalized,
            "language": row["language"],
            "line_count": row["line_count"],
            "symbol_count": len(symbol_rows),
            "functions": categorized["functions"],
            "classes": categorized["classes"],
            "methods": categorized["methods"],
            "imports": imports,
            "exports": exports,
            "docstring": row["docstring"],
        }

    def get_index_stats(self) -> Dict[str, Any]:
        """Return basic statistics for the current index."""
        with self._lock:
            if not self.store:
                return {"status": "not_loaded"}
            try:
                with self.store.connect() as conn:
                    metadata = self.store.get_metadata(conn, "index_metadata")
            except SQLiteSchemaMismatchError:
                return {"status": "not_loaded"}
            if not metadata:
                return {"status": "not_loaded"}
            return {
                "status": "loaded" if self._is_loaded else "not_loaded",
                "indexed_files": metadata.get("indexed_files", 0),
                "total_symbols": metadata.get("total_symbols", 0),
                "symbol_types": metadata.get("symbol_types", {}),
                "languages": metadata.get("languages", []),
                "project_path": metadata.get("project_path"),
                "timestamp": metadata.get("timestamp"),
            }

    def cleanup(self) -> None:
        """Reset internal state."""
        with self._lock:
            self.project_path = None
            self.index_builder = None
            self.store = None
            self.temp_dir = None
            self.index_path = None
            self._shallow_file_list = None
            self._is_loaded = False


def _hash_project_path(project_path: str) -> str:
    import hashlib

    return hashlib.md5(project_path.encode()).hexdigest()[:12]


def _compile_glob_regex(pattern: str):
    i = 0
    out = []
    special = ".^$+{}[]|()"
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c in special:
            out.append("\\" + c)
        else:
            out.append(c)
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _normalize_path(path: str) -> str:
    result = path.replace("\\\\", "/").replace("\\", "/")
    if result.startswith("./"):
        result = result[2:]
    return result


def _safe_json_loads(value: Any) -> List[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _categorize_symbols(symbol_rows) -> Dict[str, List[Dict[str, Any]]]:
    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    methods: List[Dict[str, Any]] = []

    for row in symbol_rows:
        symbol_type = row["type"]
        called_by = _safe_json_loads(row["called_by"])
        info = {
            "name": row["short_name"],
            "called_by": called_by,
            "line": row["line"],
            "end_line": row["end_line"],
            "signature": row["signature"],
            "docstring": row["docstring"],
        }

        signature = row["signature"] or ""
        if signature.startswith("def ") and "::" in signature:
            methods.append(info)
        elif signature.startswith("def "):
            functions.append(info)
        elif signature.startswith("class ") or symbol_type == "class":
            classes.append(info)
        else:
            if symbol_type == "method":
                methods.append(info)
            elif symbol_type == "class":
                classes.append(info)
            else:
                functions.append(info)

    functions.sort(key=lambda item: item.get("line") or 0)
    classes.sort(key=lambda item: item.get("line") or 0)
    methods.sort(key=lambda item: item.get("line") or 0)

    return {
        "functions": functions,
        "classes": classes,
        "methods": methods,
    }
