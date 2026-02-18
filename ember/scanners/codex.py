"""Codex session scanner — extracts project context from OpenAI Codex CLI JSONL sessions."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ember.bootstrap import BaseScanner, BootstrapMemory


class CodexScanner(BaseScanner):
    """Scans OpenAI Codex CLI session history for project context and user interactions."""

    def __init__(self, max_sessions_per_project: int = 5,
                 max_messages_per_session: int = 50):
        self.base_dir = Path.home() / ".codex" / "sessions"
        self.max_sessions_per_project = max_sessions_per_project
        self.max_messages_per_session = max_messages_per_session

    @property
    def display_name(self) -> str:
        return "Codex conversations"

    async def scan(self) -> List[BootstrapMemory]:
        if not self.base_dir.exists():
            return []

        # Find all JSONL session files
        session_files = sorted(
            self.base_dir.rglob("rollout-*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not session_files:
            return []

        # Group sessions by project (cwd)
        projects: Dict[str, List[Path]] = defaultdict(list)
        for sf in session_files:
            cwd = self._extract_cwd(sf)
            if cwd:
                projects[cwd].append(sf)
            else:
                projects["unknown"].append(sf)

        # Build memories per project
        memories: List[BootstrapMemory] = []
        for project_path, files in projects.items():
            if project_path == "unknown":
                continue
            project_name = self._project_name(project_path)
            recent = files[:self.max_sessions_per_project]
            all_messages: List[str] = []
            for sf in recent:
                msgs = self._extract_user_messages(sf)
                all_messages.extend(msgs)

            substantive = [m for m in all_messages if len(m.split()) > 5]
            if not substantive:
                continue

            sample = substantive[:20]
            content = f"Project: {project_name}\n"
            content += f"Path: {project_path}\n"
            content += "User topics and requests:\n"
            content += "\n".join(f"- {msg[:200]}" for msg in sample)

            memories.append(BootstrapMemory(
                name=f"Codex project: {project_name}",
                content=content[:2000],
                tags=["bootstrap", "codex", project_name],
                importance="context",
                source_path=str(recent[0].parent),
                source_type="codex",
            ))

        return memories

    def _extract_cwd(self, session_file: Path) -> Optional[str]:
        """Extract the project working directory from session_meta entry."""
        try:
            with open(session_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("type") == "session_meta":
                        payload = entry.get("payload", {})
                        cwd = payload.get("cwd")
                        if cwd:
                            return cwd
                    # Also check turn_context which has cwd
                    if entry.get("type") == "turn_context":
                        payload = entry.get("payload", {})
                        cwd = payload.get("cwd")
                        if cwd:
                            return cwd
        except (OSError, PermissionError):
            pass
        return None

    def _extract_user_messages(self, session_file: Path) -> List[str]:
        """Extract user messages from a Codex JSONL session file."""
        messages: List[str] = []
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

                    msg = self._parse_user_message(entry)
                    if msg:
                        messages.append(msg)
                        count += 1
        except (OSError, PermissionError):
            pass
        return messages

    def _parse_user_message(self, entry: dict) -> Optional[str]:
        """Parse a single JSONL entry for user message content."""
        entry_type = entry.get("type")
        payload = entry.get("payload", {})

        # Format 1: event_msg with user_message type
        if entry_type == "event_msg" and payload.get("type") == "user_message":
            msg = payload.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()

        # Format 2: response_item with role=user
        if entry_type == "response_item" and payload.get("role") == "user":
            content = payload.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if isinstance(text, str) and text.strip():
                            # Skip system/developer messages embedded as user content
                            if text.startswith("<") or text.startswith("#"):
                                continue
                            return text.strip()

        return None

    @staticmethod
    def _project_name(cwd: str) -> str:
        """Extract a readable project name from a path."""
        parts = Path(cwd).parts
        meaningful = [p for p in parts if p and p.lower() not in
                      {"/", "users", "home", os.environ.get("USER", "").lower(),
                       "documents", "desktop", "projects", "developer"}]
        return "-".join(meaningful[-2:]) if meaningful else Path(cwd).name
