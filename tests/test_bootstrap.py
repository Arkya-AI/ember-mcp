import json
import pytest
from pathlib import Path
from ember.bootstrap import BootstrapMemory, BaseScanner, DocumentScanner, GitRepoScanner


def test_bootstrap_memory_defaults():
    mem = BootstrapMemory(name="test", content="hello world")
    assert mem.name == "test"
    assert mem.content == "hello world"
    assert mem.tags == []
    assert mem.importance == "context"
    assert mem.source_path == ""
    assert mem.source_type == ""


def test_bootstrap_memory_with_all_fields():
    mem = BootstrapMemory(
        name="project",
        content="React app with TypeScript",
        tags=["bootstrap", "git"],
        importance="fact",
        source_path="/Users/me/project/README.md",
        source_type="git",
    )
    assert mem.importance == "fact"
    assert "git" in mem.tags


def test_base_scanner_is_abstract():
    with pytest.raises(TypeError):
        BaseScanner()


@pytest.mark.asyncio
async def test_document_scanner_finds_md_files(tmp_path):
    (tmp_path / "readme.md").write_text("# Hello World\nThis is a test document.")
    (tmp_path / "notes.txt").write_text("Some plain text notes here.")
    (tmp_path / "binary.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / ".env").write_text("SECRET_KEY=abc123")

    scanner = DocumentScanner()
    scanner.scan_dirs = [tmp_path]
    results = await scanner.scan()

    names = [m.name for m in results]
    assert "Document: readme.md" in names
    assert "Document: notes.txt" in names
    assert not any("binary" in n for n in names)
    assert not any(".env" in n for n in names)


@pytest.mark.asyncio
async def test_document_scanner_skips_large_files(tmp_path):
    (tmp_path / "big.md").write_text("x" * 20000)
    scanner = DocumentScanner(max_file_size=10240)
    scanner.scan_dirs = [tmp_path]
    results = await scanner.scan()
    assert len(results) == 0


@pytest.mark.asyncio
async def test_document_scanner_truncates_content(tmp_path):
    long_text = " ".join(["word"] * 1000)
    (tmp_path / "long.md").write_text(long_text)
    scanner = DocumentScanner(max_words=500, max_file_size=100000)
    scanner.scan_dirs = [tmp_path]
    results = await scanner.scan()
    assert len(results) == 1
    assert len(results[0].content.split()) == 500


@pytest.mark.asyncio
async def test_document_scanner_respects_depth(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.md").write_text("Too deep")
    (tmp_path / "shallow.md").write_text("Shallow")

    scanner = DocumentScanner(max_depth=1)
    scanner.scan_dirs = [tmp_path]
    results = await scanner.scan()
    names = [m.name for m in results]
    assert "Document: shallow.md" in names
    assert "Document: deep.md" not in names


@pytest.mark.asyncio
async def test_git_scanner_reads_readme(tmp_path):
    repo = tmp_path / "my-project"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "README.md").write_text("# My Project\nA cool thing")

    scanner = GitRepoScanner(max_days=9999)
    scanner.scan_dirs = [tmp_path]
    readme = scanner._read_readme(repo, "my-project")
    assert readme is not None
    assert "My Project" in readme.content
    assert readme.source_type == "git"


@pytest.mark.asyncio
async def test_git_scanner_detects_tech_stack(tmp_path):
    repo = tmp_path / "my-project"
    repo.mkdir()
    (repo / "package.json").write_text('{"name":"my-project","dependencies":{"react":"^18","express":"^4"}}')

    scanner = GitRepoScanner()
    stack = scanner._detect_tech_stack(repo, "my-project")
    assert stack is not None
    assert "javascript" in stack.content
    assert "react" in stack.content
    assert stack.importance == "fact"


@pytest.mark.asyncio
async def test_git_scanner_no_readme_returns_none(tmp_path):
    repo = tmp_path / "empty-project"
    repo.mkdir()
    scanner = GitRepoScanner()
    assert scanner._read_readme(repo, "empty-project") is None


@pytest.mark.asyncio
async def test_git_scanner_no_tech_stack_returns_none(tmp_path):
    repo = tmp_path / "no-stack"
    repo.mkdir()
    scanner = GitRepoScanner()
    assert scanner._detect_tech_stack(repo, "no-stack") is None


from ember.bootstrap import BootstrapPipeline


@pytest.mark.asyncio
async def test_pipeline_registers_and_runs_scanners():
    """Test that pipeline runs scanners and collects results."""

    class FakeScanner(BaseScanner):
        @property
        def display_name(self):
            return "fake"
        async def scan(self):
            return [
                BootstrapMemory(name="test1", content="Hello world testing"),
                BootstrapMemory(name="test2", content="Goodbye world testing"),
            ]

    class FakeEngine:
        def embed(self, text):
            import numpy as np
            return np.random.rand(1, 384).astype("float32")
        def assign_cell(self, vec):
            return 0
        def add_vector(self, faiss_id, vec):
            pass
        def save_index(self):
            pass

    class FakeStorage:
        def __init__(self):
            self.saved = []
        async def save_ember(self, ember):
            self.saved.append(ember)
            return len(self.saved) - 1
        async def update_region(self, cell_id, vitality, shadow_accum):
            pass

    engine = FakeEngine()
    storage = FakeStorage()
    pipeline = BootstrapPipeline(engine, storage)
    pipeline.register_scanner(FakeScanner())

    result = await pipeline.run()
    assert result["total_scanned"] == 2
    assert result["total_stored"] == 2
    assert len(storage.saved) == 2


@pytest.mark.asyncio
async def test_pipeline_dry_run_stores_nothing():
    class FakeScanner(BaseScanner):
        @property
        def display_name(self):
            return "fake"
        async def scan(self):
            return [BootstrapMemory(name="test", content="Some content here")]

    class FakeEngine:
        def embed(self, text):
            import numpy as np
            return np.random.rand(1, 384).astype("float32")

    pipeline = BootstrapPipeline(FakeEngine(), None)
    pipeline.register_scanner(FakeScanner())

    result = await pipeline.run(dry_run=True)
    assert result["total_scanned"] == 1
    assert result["total_stored"] == 0
    assert result.get("dry_run") is True


@pytest.mark.asyncio
async def test_pipeline_deduplication():
    class FakeEngine:
        def embed(self, text):
            import numpy as np
            # Return same vector for similar content so they get deduplicated
            if "hello" in text.lower():
                vec = np.ones((1, 384), dtype="float32")
            else:
                vec = np.zeros((1, 384), dtype="float32")
                vec[0, 0] = 1.0
            return vec

    engine = FakeEngine()
    pipeline = BootstrapPipeline(engine, None)

    memories = [
        BootstrapMemory(name="mem1", content="hello world"),
        BootstrapMemory(name="mem2", content="hello there world"),
        BootstrapMemory(name="mem3", content="completely different content"),
    ]

    unique = pipeline._deduplicate(memories)
    # The two "hello" ones should be deduped, "completely different" should remain
    assert len(unique) == 2


# ── CodexScanner tests ─────────────────────────────────────────────────

from ember.scanners.codex import CodexScanner


def test_codex_scanner_display_name():
    scanner = CodexScanner()
    assert scanner.display_name == "Codex conversations"


@pytest.mark.asyncio
async def test_codex_scanner_scans_sessions(tmp_path):
    scanner = CodexScanner()
    scanner.base_dir = tmp_path

    session_dir = tmp_path / "2026" / "02" / "17"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "rollout-test.jsonl"

    lines = [
        json.dumps({"timestamp": "2026-02-17T00:00:00Z", "type": "session_meta",
                     "payload": {"id": "test", "cwd": "/Users/me/myproject"}}),
        json.dumps({"timestamp": "2026-02-17T00:00:01Z", "type": "event_msg",
                     "payload": {"type": "user_message",
                                 "message": "implement the login page with OAuth support",
                                 "images": []}}),
        json.dumps({"timestamp": "2026-02-17T00:00:02Z", "type": "event_msg",
                     "payload": {"type": "user_message",
                                 "message": "add error handling to the API endpoints",
                                 "images": []}}),
    ]
    session_file.write_text("\n".join(lines))

    results = await scanner.scan()
    assert len(results) == 1
    assert "Codex project" in results[0].name
    assert "myproject" in results[0].content
    assert "login page" in results[0].content
    assert results[0].source_type == "codex"


@pytest.mark.asyncio
async def test_codex_scanner_empty_dir(tmp_path):
    scanner = CodexScanner()
    scanner.base_dir = tmp_path
    results = await scanner.scan()
    assert results == []


@pytest.mark.asyncio
async def test_codex_scanner_skips_short_messages(tmp_path):
    scanner = CodexScanner()
    scanner.base_dir = tmp_path

    session_dir = tmp_path / "2026" / "02" / "17"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "rollout-short.jsonl"

    lines = [
        json.dumps({"timestamp": "2026-02-17T00:00:00Z", "type": "session_meta",
                     "payload": {"id": "short", "cwd": "/Users/me/proj"}}),
        json.dumps({"timestamp": "2026-02-17T00:00:01Z", "type": "event_msg",
                     "payload": {"type": "user_message", "message": "ok", "images": []}}),
        json.dumps({"timestamp": "2026-02-17T00:00:02Z", "type": "event_msg",
                     "payload": {"type": "user_message", "message": "yes please", "images": []}}),
    ]
    session_file.write_text("\n".join(lines))

    results = await scanner.scan()
    assert results == []


# ── CopilotScanner tests ───────────────────────────────────────────────

from ember.scanners.copilot import CopilotScanner


def test_copilot_scanner_display_name():
    scanner = CopilotScanner()
    assert scanner.display_name == "GitHub Copilot"


@pytest.mark.asyncio
async def test_copilot_scanner_empty_when_no_data():
    scanner = CopilotScanner()
    results = await scanner.scan()
    # No real Copilot data in test environment — should return empty
    assert isinstance(results, list)


def test_copilot_scanner_scans_json_files(tmp_path):
    scanner = CopilotScanner()
    chat_data = [
        {"role": "user", "text": "How do I implement binary search in Python?"},
        {"role": "assistant", "text": "Here is an implementation..."},
        {"role": "user", "text": "Can you add error handling for empty lists?"},
    ]
    (tmp_path / "chat.json").write_text(json.dumps(chat_data))

    results = scanner._scan_json_dir(tmp_path, "test")
    assert len(results) == 1
    assert "binary search" in results[0].content
    assert results[0].source_type == "copilot"


def test_copilot_scanner_extract_user_prompts():
    scanner = CopilotScanner()
    data = {
        "messages": [
            {"role": "user", "content": "test prompt"},
            {"role": "assistant", "content": "response"},
        ]
    }
    prompts = scanner._extract_user_prompts(data)
    assert "test prompt" in prompts
    assert "response" not in prompts
