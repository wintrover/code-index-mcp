"""Tests for Content-Addressable Lazy Index with Integrity Levels."""

from __future__ import annotations

import hashlib
import os
import tempfile

import pytest

from code_index_mcp.indexing.sqlite_store import SQLiteIndexStore, SCHEMA_VERSION
from code_index_mcp.indexing.sqlite_index_manager import SQLiteIndexManager
from code_index_mcp.indexing.sqlite_index_builder import SQLiteIndexBuilder


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project with sample Python files."""
    (tmp_path / "sample.py").write_text(
        'def hello():\n    print("hello")\n\n\ndef world():\n    hello()\n'
    )
    (tmp_path / "main.py").write_text(
        "from sample import hello\n\n\ndef run():\n    hello()\n"
    )
    return tmp_path


@pytest.fixture
def store(tmp_path):
    """Create a fresh SQLite store with initialized schema."""
    db_path = str(tmp_path / "test.db")
    s = SQLiteIndexStore(db_path)
    s.initialize_schema()
    return s


@pytest.fixture
def manager(tmp_path, store):
    """Create a SQLiteIndexManager wired to a temporary project."""
    mgr = SQLiteIndexManager()
    mgr.set_project_path(str(tmp_path))
    # Override the store so our fixture's store (with initialized schema) is used.
    mgr.store = store
    return mgr


# ── Schema tests ──────────────────────────────────────────────────────────


class TestSchemaVersion:
    """Test schema version and new columns exist."""

    def test_schema_version_is_4(self):
        assert SCHEMA_VERSION == 4

    def test_fresh_db_has_current_schema_version(self, store):
        with store.connect() as conn:
            version = store.get_metadata(conn, "schema_version")
        assert version == SCHEMA_VERSION

    def test_files_table_has_content_hash_column(self, store):
        with store.connect() as conn:
            columns = [c["name"] for c in conn.execute("PRAGMA table_info(files)").fetchall()]
        assert "content_hash" in columns

    def test_files_table_has_integrity_level_column(self, store):
        with store.connect() as conn:
            columns = [c["name"] for c in conn.execute("PRAGMA table_info(files)").fetchall()]
        assert "integrity_level" in columns

    def test_default_integrity_level_is_global_linked(self, store):
        """Files inserted without explicit integrity_level default to GLOBAL_LINKED."""
        with store.connect() as conn:
            # Omit integrity_level to trigger the column DEFAULT
            conn.execute("INSERT INTO files (path) VALUES (?)", ("test.py",))
            row = conn.execute(
                "SELECT integrity_level FROM files WHERE path = 'test.py'"
            ).fetchone()
            assert row["integrity_level"] == "GLOBAL_LINKED"


# ── Lazy parse tests ──────────────────────────────────────────────────────


class TestLazyParse:
    """Test lazy parse and cache via get_file_summary."""

    def test_lazy_parse_returns_local_ast(self, manager, tmp_project):
        """get_file_summary on an uncached file should return LOCAL_AST metadata."""
        result = manager.get_file_summary("sample.py")

        assert result is not None
        assert result["meta"]["integrity_level"] == "LOCAL_AST"
        assert result["meta"]["content_hash"] is not None
        assert result["capabilities"]["cross_references"] == "REQUIRES_DEEP_INDEX"
        assert result["capabilities"]["local_symbols"] == "AVAILABLE"

    def test_lazy_parse_caches_in_store(self, manager, tmp_project):
        """After lazy parse the file row should be persisted with LOCAL_AST."""
        manager.get_file_summary("sample.py")

        with manager.store.connect() as conn:
            row = conn.execute(
                "SELECT integrity_level, content_hash FROM files WHERE path = ?",
                ("sample.py",),
            ).fetchone()
        assert row is not None
        assert row["integrity_level"] == "LOCAL_AST"
        assert row["content_hash"] is not None

    def test_cache_hit_same_hash(self, manager, tmp_project):
        """Second call with same content returns cached result without re-parse."""
        result1 = manager.get_file_summary("sample.py")
        result2 = manager.get_file_summary("sample.py")

        assert result1 is not None
        assert result2 is not None
        assert result1["meta"]["content_hash"] == result2["meta"]["content_hash"]
        assert result2["meta"]["integrity_level"] == "LOCAL_AST"

    def test_cache_invalidation_on_content_change(self, manager, tmp_project):
        """Changed file content → new hash → re-parsed."""
        result1 = manager.get_file_summary("sample.py")
        old_hash = result1["meta"]["content_hash"]

        # Modify the file
        (tmp_project / "sample.py").write_text('def hello():\n    print("modified!")\n')

        result2 = manager.get_file_summary("sample.py")
        assert result2["meta"]["content_hash"] != old_hash
        assert result2["meta"]["integrity_level"] == "LOCAL_AST"

    def test_nonexistent_file_returns_none(self, manager, tmp_project):
        assert manager.get_file_summary("nonexistent.py") is None

    def test_lazy_parse_detects_symbols(self, manager, tmp_project):
        """Parsed result should contain the two functions in sample.py."""
        result = manager.get_file_summary("sample.py")
        fn_names = [f["name"] for f in result["functions"]]
        assert "hello" in fn_names
        assert "world" in fn_names

    def test_binary_file_returns_none(self, manager, tmp_project):
        """Binary files should be skipped."""
        (tmp_project / "data.bin").write_bytes(b"\x00\x01\x02\x03")
        assert manager.get_file_summary("data.bin") is None


# ── Promotion tests ──────────────────────────────────────────────────────


class TestPromotion:
    """Test LOCAL_AST → GLOBAL_LINKED promotion via build_index."""

    def test_promote_local_ast_to_global(self, manager, tmp_project):
        """build_index should promote existing LOCAL_AST files to GLOBAL_LINKED."""
        # Create LOCAL_AST data via lazy parse
        manager.get_file_summary("sample.py")
        manager.get_file_summary("main.py")

        # Verify LOCAL_AST
        with manager.store.connect() as conn:
            local_count = conn.execute(
                "SELECT COUNT(*) FROM files WHERE integrity_level = 'LOCAL_AST'"
            ).fetchone()[0]
        assert local_count == 2

        # Build index (non-force) → should promote
        builder = SQLiteIndexBuilder(str(tmp_project), manager.store)
        builder.build_index(force_rebuild=False)

        # Verify all promoted to GLOBAL_LINKED
        with manager.store.connect() as conn:
            local_count = conn.execute(
                "SELECT COUNT(*) FROM files WHERE integrity_level = 'LOCAL_AST'"
            ).fetchone()[0]
            global_count = conn.execute(
                "SELECT COUNT(*) FROM files WHERE integrity_level = 'GLOBAL_LINKED'"
            ).fetchone()[0]
        assert local_count == 0
        assert global_count == 2

    def test_force_rebuild_clears_and_rebuilds(self, manager, tmp_project):
        """force_rebuild=True should discard existing data and rebuild from scratch."""
        # Create LOCAL_AST data
        manager.get_file_summary("sample.py")

        builder = SQLiteIndexBuilder(str(tmp_project), manager.store)
        stats = builder.build_index(force_rebuild=True)

        assert stats["files"] >= 2  # sample.py + main.py
        with manager.store.connect() as conn:
            rows = conn.execute("SELECT integrity_level FROM files").fetchall()
            for row in rows:
                assert row["integrity_level"] == "GLOBAL_LINKED"

    def test_promotion_preserves_content_hash(self, manager, tmp_project):
        """Promoted files should retain their content_hash."""
        manager.get_file_summary("sample.py")

        with manager.store.connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM files WHERE path = 'sample.py'"
            ).fetchone()
            hash_before = row["content_hash"]

        builder = SQLiteIndexBuilder(str(tmp_project), manager.store)
        builder.build_index(force_rebuild=False)

        with manager.store.connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM files WHERE path = 'sample.py'"
            ).fetchone()
            hash_after = row["content_hash"]

        assert hash_before is not None
        assert hash_after is not None
        assert hash_before == hash_after

    def test_no_local_ast_skips_promotion(self, manager, tmp_project):
        """When no LOCAL_AST files exist, build_index proceeds with full rebuild."""
        builder = SQLiteIndexBuilder(str(tmp_project), manager.store)
        stats = builder.build_index(force_rebuild=False)

        assert stats["files"] >= 2
        with manager.store.connect() as conn:
            local = conn.execute(
                "SELECT COUNT(*) FROM files WHERE integrity_level = 'LOCAL_AST'"
            ).fetchone()[0]
        assert local == 0


# ── Store-level tests ────────────────────────────────────────────────────


class TestStoreIntegrity:
    """Test SQLiteIndexStore upsert_file_with_integrity and get_file_with_hash."""

    def test_upsert_and_get_with_hash(self, store):
        """Round-trip upsert_file_with_integrity → get_file_with_hash."""
        from types import SimpleNamespace

        file_info = SimpleNamespace(
            language="python",
            line_count=4,
            imports=None,
            exports=None,
            package=None,
            docstring=None,
        )
        symbols = [
            SimpleNamespace(
                symbol_id="sample.py::hello",
                type="function",
                line=1,
                end_line=2,
                signature="def hello()",
                docstring=None,
                called_by=[],
                short_name="hello",
            )
        ]
        content_hash = hashlib.sha256(b"test content").hexdigest()

        store.upsert_file_with_integrity(
            path="sample.py",
            file_info=file_info,
            symbols=symbols,
            content_hash=content_hash,
            integrity_level="LOCAL_AST",
        )

        result = store.get_file_with_hash("sample.py", content_hash)
        assert result is not None
        assert result["integrity_level"] == "LOCAL_AST"
        assert result["content_hash"] == content_hash

    def test_get_file_with_hash_mismatch_returns_none(self, store):
        """get_file_with_hash returns None when hash doesn't match."""
        result = store.get_file_with_hash("nonexistent.py", "abc123")
        assert result is None


# ── _read_file_safe tests ────────────────────────────────────────────────


class TestReadFileSafe:
    """Test _read_file_safe helper on the manager."""

    def test_read_normal_file(self, manager, tmp_project):
        content = manager._read_file_safe(str(tmp_project / "sample.py"))
        assert content is not None
        assert "def hello" in content

    def test_read_binary_returns_none(self, manager, tmp_project):
        (tmp_project / "bin.dat").write_bytes(b"\x00\x00")
        content = manager._read_file_safe(str(tmp_project / "bin.dat"))
        assert content is None

    def test_read_nonexistent_returns_none(self, manager):
        content = manager._read_file_safe("/no/such/file.py")
        assert content is None
