#!/usr/bin/env python3
"""
End-to-end test: Index the real Axiom Nim project and exercise all major MCP tools.

This tests:
  1. Shallow index (file discovery) on 1600+ .nim files
  2. Deep index (symbol extraction) with NimParsingStrategy
  3. File summary (line count, symbols, imports)
  4. Symbol body retrieval for procs, funcs, types
  5. Code search via the search service
"""

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Project path
# ---------------------------------------------------------------------------
PROJECT = Path("/home/wintrover/바탕화면/Axiom_CLI/Axiom")
assert PROJECT.exists(), f"Project not found: {PROJECT}"

nim_files = list(PROJECT.rglob("*.nim"))
print(f"\n{'='*70}")
print(f"  Nim project: {PROJECT}")
print(f"  .nim files discovered on disk: {len(nim_files)}")
print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# 2. Shallow index  (SQLiteIndexManager)
# ---------------------------------------------------------------------------
from code_index_mcp.indexing.sqlite_index_manager import SQLiteIndexManager

t0 = time.time()
manager = SQLiteIndexManager()
ok = manager.set_project_path(str(PROJECT))
print(f"[shallow] set_project_path -> {ok}")

ok = manager.build_index(max_workers=4, timeout=300)
elapsed_build = time.time() - t0
print(f"[shallow] build_index     -> {ok}  ({elapsed_build:.1f}s)")
if not ok:
    print("FATAL: build_index failed")
    sys.exit(1)

ok = manager.load_index()
print(f"[shallow] load_index      -> {ok}")

# Build the shallow index too (JSON-based file list, separate from SQLite)
t_shallow = time.time()
shallow_ok = manager.build_shallow_index()
elapsed_shallow = time.time() - t_shallow
print(f"[shallow] build_shallow   -> {shallow_ok}  ({elapsed_shallow:.1f}s)")
ok = manager.load_shallow_index()
print(f"[shallow] load_shallow    -> {ok}")

# Index stats
stats = manager.get_index_stats()
print(f"[shallow] index stats     : {json.dumps(stats, indent=2)}")

# Find files pattern (uses shallow index)
# Note: "*.nim" only matches root-level; use "**/*.nim" for recursive match
all_files = manager.find_files("*")
nim_in_index = [f for f in all_files if f.endswith(".nim")]
found = manager.find_files("**/*.nim")
print(f"[find_files] *      -> {len(all_files)} files total")
print(f"[find_files] **/*.nim -> {len(found)} files")
print(f"[find_files] (filtered) -> {len(nim_in_index)} .nim files")
assert len(found) > 0 or len(nim_in_index) > 0, "No .nim files found via find_files"
# Use whichever list is non-empty for downstream tests
sample_pool = found if found else nim_in_index

# ---------------------------------------------------------------------------
# 3. Pick a few .nim files and test get_file_summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  FILE SUMMARIES")
print(f"{'='*70}")

# Pick 5 .nim files from the discovered pool for summary
sample_files = sample_pool[:5]
summaries_found = 0

for rel_path in sample_files:
    summary = manager.get_file_summary(rel_path)
    if summary is None:
        print(f"  [skip] {rel_path} -> no summary")
        continue

    summaries_found += 1
    lang = summary.get("language", "?")
    lines = summary.get("line_count", 0)
    sym_count = summary.get("symbol_count", 0)
    funcs = summary.get("functions", [])
    methods = summary.get("methods", [])
    classes = summary.get("classes", [])
    imports = summary.get("imports", [])

    print(f"\n  File: {rel_path}")
    print(f"    language   : {lang}")
    print(f"    lines      : {lines}")
    print(f"    symbols    : {sym_count}")
    print(f"    functions  : {[s['name'] for s in funcs]}")
    print(f"    methods    : {[s['name'] for s in methods]}")
    print(f"    classes    : {[s['name'] for s in classes]}")
    print(f"    imports    : {imports[:5]}{'...' if len(imports) > 5 else ''}")

    if lang != "nim":
        print(f"    WARNING: expected language 'nim', got '{lang}'")

print(f"\n  Summaries retrieved: {summaries_found}/{len(sample_files)}")
assert summaries_found > 0, "No file summaries retrieved"

# ---------------------------------------------------------------------------
# 4. Test get_symbol_body via CodeIntelligenceService
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  SYMBOL BODY RETRIEVAL")
print(f"{'='*70}")

from code_index_mcp.services import code_intelligence_service
from code_index_mcp.project_settings import ProjectSettings

class _DummyLifespanCtx:
    def __init__(self, base_path):
        self.base_path = base_path
        self.settings = ProjectSettings(base_path, skip_load=False)
        self.file_watcher_service = None

class _DummyReqCtx:
    def __init__(self, base_path):
        self.lifespan_context = _DummyLifespanCtx(base_path)
        self.session = None
        self.meta = None

class _DummyCtx:
    def __init__(self, base_path):
        self.request_context = _DummyReqCtx(base_path)

# Patch get_index_manager to return our manager
original_get_index_manager = code_intelligence_service.get_index_manager
code_intelligence_service.get_index_manager = lambda: manager

ci_svc = code_intelligence_service.CodeIntelligenceService(ctx=_DummyCtx(str(PROJECT)))

body_results = 0
for rel_path in sample_files:
    summary = manager.get_file_summary(rel_path)
    if summary is None:
        continue

    # Try to get body of first function, method, or class
    target_symbol = None
    symbol_type = None
    if summary.get("functions"):
        target_symbol = summary["functions"][0]["name"]
        symbol_type = "function"
    elif summary.get("methods"):
        target_symbol = summary["methods"][0]["name"]
        symbol_type = "method"
    elif summary.get("classes"):
        target_symbol = summary["classes"][0]["name"]
        symbol_type = "class"

    if target_symbol is None:
        print(f"  [skip] {rel_path} -> no symbols found")
        continue

    body = ci_svc.get_symbol_body(rel_path, target_symbol)
    if body is None:
        print(f"  [warn] get_symbol_body({rel_path}, {target_symbol}) -> None")
        continue

    body_results += 1
    status = body.get("status", "?")
    code_preview = body.get("code", "")[:120].replace("\n", "\u21b5")
    sig = body.get("signature", "")
    line = body.get("line", "?")
    end_line = body.get("end_line", "?")
    found_type = body.get("type", "?")

    print(f"\n  Symbol : {target_symbol} (type={found_type})")
    print(f"  File   : {rel_path}")
    print(f"  Status : {status}")
    print(f"  Lines  : {line}-{end_line}")
    if sig:
        print(f"  Sig    : {sig[:100]}")
    print(f"  Code   : {code_preview}...")

# Restore original
code_intelligence_service.get_index_manager = original_get_index_manager

print(f"\n  Symbol bodies retrieved: {body_results}")
assert body_results > 0, "No symbol bodies retrieved"

# ---------------------------------------------------------------------------
# 5. Search for Nim-specific patterns
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  CODE SEARCH")
print(f"{'='*70}")

from code_index_mcp.services.search_service import SearchService

search_ctx = _DummyCtx(str(PROJECT))
svc = SearchService(search_ctx)

search_patterns = [
    ("proc ", "*.nim"),
    ("func ", "*.nim"),
    ("type ", "*.nim"),
    ("import std", "*.nim"),
]

for pattern, file_pat in search_patterns:
    t2 = time.time()
    result = svc.search_code(
        pattern=pattern,
        case_sensitive=True,
        file_pattern=file_pat,
        start_index=0,
        max_results=5,
    )
    elapsed = time.time() - t2
    total = result.get("total_matches", 0) if isinstance(result, dict) else "?"
    matches = result.get("matches", []) if isinstance(result, dict) else []
    print(f"\n  Pattern: '{pattern}' in {file_pat}")
    print(f"    Total matches : {total}")
    print(f"    Time          : {elapsed:.2f}s")
    for m in matches[:3]:
        fp = m.get("file", "?")
        ln = m.get("line", "?")
        text = m.get("text", "")[:80]
        print(f"    [{fp}:{ln}] {text}")
    if not matches:
        print(f"    Raw result: {result}")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  E2E TEST SUMMARY")
print(f"{'='*70}")
print(f"  Project              : {PROJECT}")
print(f"  .nim files on disk   : {len(nim_files)}")
print(f"  .nim files in index  : {len(sample_pool)}")
print(f"  Index build time     : {elapsed_build:.1f}s")
print(f"  File summaries       : {summaries_found} OK")
print(f"  Symbol bodies        : {body_results} OK")
print(f"  Search patterns      : {len(search_patterns)} tested")
print(f"{'='*70}")
print(f"  ALL CHECKS PASSED")
print(f"{'='*70}\n")
