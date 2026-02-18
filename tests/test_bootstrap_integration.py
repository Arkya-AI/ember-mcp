"""Integration test: runs the full bootstrap pipeline end-to-end."""
import json
import pytest
from pathlib import Path

from ember.models import EmberConfig
from ember.core import VectorEngine
from ember.storage import StorageManager
from ember.bootstrap import BootstrapPipeline, DocumentScanner
from ember.scanners.claude_code import ClaudeCodeScanner


@pytest.fixture
def temp_env(tmp_path):
    """Create temp environment with fake data sources and Ember storage."""
    data_dir = tmp_path / "ember_data"
    data_dir.mkdir()

    # Fake documents
    docs_dir = tmp_path / "Documents"
    docs_dir.mkdir()
    (docs_dir / "notes.md").write_text("# Project Notes\nDecided to use PostgreSQL for the database. This is a detailed decision record.")
    (docs_dir / "todo.txt").write_text("TODO: Set up CI/CD pipeline with GitHub Actions. Need to configure workflows.")

    # Fake Claude Code sessions
    claude_dir = tmp_path / "claude" / "projects" / "-test-project"
    claude_dir.mkdir(parents=True)
    session = claude_dir / "session-1.jsonl"
    session.write_text("\n".join([
        json.dumps({"type": "user", "cwd": "/test/project", "message": {"role": "user", "content": "Build a REST API with Express and TypeScript for the backend"}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": "Add rate limiting middleware to protect the API endpoints"}}),
    ]))

    return {
        "data_dir": data_dir,
        "docs_dir": docs_dir,
        "claude_dir": tmp_path / "claude" / "projects",
    }


@pytest.mark.asyncio
async def test_full_pipeline(temp_env):
    """End-to-end: scan docs + Claude Code sessions, deduplicate, store as embers."""
    config = EmberConfig(data_dir=temp_env["data_dir"])
    engine = VectorEngine(config)
    storage = StorageManager(config)
    await storage.init_db()

    pipeline = BootstrapPipeline(engine, storage)

    doc_scanner = DocumentScanner()
    doc_scanner.scan_dirs = [temp_env["docs_dir"]]
    pipeline.register_scanner(doc_scanner)

    claude_scanner = ClaudeCodeScanner()
    claude_scanner.projects_dir = temp_env["claude_dir"]
    pipeline.register_scanner(claude_scanner)

    result = await pipeline.run()

    assert result["total_scanned"] >= 3  # 2 docs + 1 claude project
    assert result["total_stored"] >= 2   # After dedup
    assert result["total_stored"] <= result["total_scanned"]

    # Verify embers are actually persisted
    embers = await storage.list_embers()
    assert len(embers) >= 2
    assert all(a.source == "bootstrap" for a in embers)
    assert all("bootstrap" in a.tags for a in embers)


@pytest.mark.asyncio
async def test_dry_run_stores_nothing(temp_env):
    """Dry run scans but stores nothing."""
    config = EmberConfig(data_dir=temp_env["data_dir"])
    engine = VectorEngine(config)
    storage = StorageManager(config)
    await storage.init_db()

    pipeline = BootstrapPipeline(engine, storage)
    doc_scanner = DocumentScanner()
    doc_scanner.scan_dirs = [temp_env["docs_dir"]]
    pipeline.register_scanner(doc_scanner)

    result = await pipeline.run(dry_run=True)

    assert result["total_scanned"] >= 2
    assert result["total_stored"] == 0
    assert result.get("dry_run") is True

    embers = await storage.list_embers()
    assert len(embers) == 0
