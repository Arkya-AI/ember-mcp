"""Claude Code session scanner — extracts project context from Claude Code JSONL sessions."""

import json
import os
from pathlib import Path
from typing import List, Optional

from ember.bootstrap import BaseScanner, BootstrapMemory


class ClaudeCodeScanner(BaseScanner):
    """Scans Claude Code session history for project context and user interactions."""

    def __init__(self, max_sessions_per_project: int = 5,
                 max_messages_per_session: int = 50):
        self.projects_dir = Path.home() / ".claude" / "projects"
        self.max_sessions_per_project = max_sessions_per_project
        self.max_messages_per_session = max_messages_per_session

    @property
    def display_name(self) -> str:
        return "Claude Code conversations"

    async def scan(self) -> List[BootstrapMemory]:
        if not self.projects_dir.exists():
            return []
        memories: List[BootstrapMemory] = []
        for project_dir in self._iter_project_dirs():
            project_name = self._decode_project_name(project_dir.name)
            project_memories = self._scan_project(project_dir, project_name)
            memories.extend(project_memories)
        return memories

    def _iter_project_dirs(self) -> List[Path]:
        try:
            return sorted(
                [d for d in self.projects_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )
        except (PermissionError, OSError):
            return []

    def _decode_project_name(self, encoded: str) -> str:
        """Decode directory name like -Users-poornamac-Documents-Project to readable name."""
        parts = encoded.split("-")
        meaningful = [p for p in parts if p and p.lower() not in
                      {"users", "home", os.environ.get("USER", "").lower(),
                       "documents", "desktop", "projects", "developer"}]
        return "-".join(meaningful[-2:]) if meaningful else encoded

    def _scan_project(self, project_dir: Path, project_name: str) -> List[BootstrapMemory]:
        memories = []
        session_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:self.max_sessions_per_project]

        all_user_messages = []
        project_path = None

        for session_file in session_files:
            messages, cwd = self._parse_session(session_file)
            all_user_messages.extend(messages)
            if cwd and not project_path:
                project_path = cwd

        if not all_user_messages:
            return memories

        substantive = [m for m in all_user_messages if len(m.split()) > 5]
        if substantive:
            sample = substantive[:20]
            content = f"Project: {project_name}\n"
            if project_path:
                content += f"Path: {project_path}\n"
            content += "User topics and requests:\n"
            content += "\n".join(f"- {msg[:200]}" for msg in sample)

            memories.append(BootstrapMemory(
                name=f"Claude Code project: {project_name}",
                content=content[:2000],
                tags=["bootstrap", "claude-code", project_name],
                importance="context",
                source_path=str(project_dir),
                source_type="claude-code",
            ))

        return memories

    def _parse_session(self, session_file: Path) -> tuple:
        """Parse a JSONL session file. Returns (user_messages, cwd)."""
        messages = []
        cwd = None
        try:
            with open(session_file, "r", encoding="utf-8", errors="ignore") as f:
                count = 0
                for line in f:
                    if count >= self.max_messages_per_session:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "cwd" in entry and not cwd:
                        cwd = entry["cwd"]
                    if entry.get("type") == "user":
                        msg = entry.get("message", {})
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                            if isinstance(content, str) and content.strip():
                                messages.append(content.strip())
                                count += 1
        except (OSError, PermissionError):
            pass
        return messages, cwd
