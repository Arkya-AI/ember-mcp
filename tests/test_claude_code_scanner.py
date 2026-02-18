import json
import pytest
from pathlib import Path
from ember.scanners.claude_code import ClaudeCodeScanner


@pytest.mark.asyncio
async def test_claude_code_scanner_parses_sessions(tmp_path):
    project_dir = tmp_path / "projects" / "-Users-test-Documents-MyProject"
    project_dir.mkdir(parents=True)

    session_file = project_dir / "abc-123.jsonl"
    lines = [
        json.dumps({"type": "queue-operation", "operation": "dequeue", "sessionId": "abc-123"}),
        json.dumps({"type": "user", "cwd": "/Users/test/Documents/MyProject", "message": {"role": "user", "content": "Help me build a REST API with FastAPI"}, "sessionId": "abc-123"}),
        json.dumps({"type": "user", "cwd": "/Users/test/Documents/MyProject", "message": {"role": "user", "content": "Add authentication with JWT tokens"}, "sessionId": "abc-123"}),
        json.dumps({"type": "user", "message": {"role": "user", "content": "ok"}, "sessionId": "abc-123"}),
    ]
    session_file.write_text("\n".join(lines))

    scanner = ClaudeCodeScanner()
    scanner.projects_dir = tmp_path / "projects"
    results = await scanner.scan()

    assert len(results) >= 1
    mem = results[0]
    assert "claude-code" in mem.tags
    assert "REST API" in mem.content or "FastAPI" in mem.content


def test_claude_code_decode_project_name():
    scanner = ClaudeCodeScanner()
    result = scanner._decode_project_name("-Users-testuser-Documents-Projects-MyProject")
    assert "MyProject" in result or "Projects" in result


@pytest.mark.asyncio
async def test_claude_code_scanner_empty_dir(tmp_path):
    scanner = ClaudeCodeScanner()
    scanner.projects_dir = tmp_path / "nonexistent"
    results = await scanner.scan()
    assert results == []


@pytest.mark.asyncio
async def test_claude_code_scanner_skips_short_messages(tmp_path):
    project_dir = tmp_path / "projects" / "-Users-test-Project"
    project_dir.mkdir(parents=True)

    session_file = project_dir / "session.jsonl"
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": "ok"}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": "yes"}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": "thanks"}}),
    ]
    session_file.write_text("\n".join(lines))

    scanner = ClaudeCodeScanner()
    scanner.projects_dir = tmp_path / "projects"
    results = await scanner.scan()
    # All messages are short (<5 words), so no substantive content = no memory
    assert len(results) == 0
