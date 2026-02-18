"""GitHub Copilot scanner — extracts chat history from VS Code, Cursor, and JetBrains."""

import json
import os
import platform
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ember.bootstrap import BaseScanner, BootstrapMemory


class CopilotScanner(BaseScanner):
    """Scans GitHub Copilot chat history across VS Code, Cursor, and JetBrains IDEs."""

    def __init__(self, max_chars_per_memory: int = 2000):
        self.max_chars_per_memory = max_chars_per_memory

    @property
    def display_name(self) -> str:
        return "GitHub Copilot"

    async def scan(self) -> List[BootstrapMemory]:
        memories: List[BootstrapMemory] = []

        # 1. VS Code workspace storage (SQLite state.vscdb)
        vscode_base = self._get_ide_path("Code")
        if vscode_base:
            memories.extend(self._scan_workspace_storage(
                vscode_base / "User" / "workspaceStorage"))
            memories.extend(self._scan_global_storage(
                vscode_base / "User" / "globalStorage" / "github.copilot-chat"))

        # 2. Cursor workspace storage
        cursor_base = self._get_ide_path("Cursor")
        if cursor_base:
            memories.extend(self._scan_workspace_storage(
                cursor_base / "User" / "workspaceStorage"))
            memories.extend(self._scan_global_storage(
                cursor_base / "User" / "globalStorage" / "github.copilot-chat"))

        # 3. JetBrains Copilot
        jb_base = self._get_jetbrains_path()
        if jb_base and jb_base.exists():
            for product_dir in self._safe_iterdir(jb_base):
                if product_dir.is_dir():
                    copilot_dir = product_dir / "copilot"
                    memories.extend(self._scan_json_dir(copilot_dir))

        return memories

    def _get_ide_path(self, app_name: str) -> Optional[Path]:
        """Return the base config path for a VS Code-like IDE."""
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library" / "Application Support" / app_name
        elif system == "Linux":
            return Path.home() / ".config" / app_name
        elif system == "Windows":
            appdata = os.environ.get("APPDATA")
            if appdata:
                return Path(appdata) / app_name
        return None

    def _get_jetbrains_path(self) -> Optional[Path]:
        """Return the base JetBrains config path."""
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library" / "Application Support" / "JetBrains"
        elif system == "Linux":
            p = Path.home() / ".local" / "share" / "JetBrains"
            return p if p.exists() else Path.home() / ".config" / "JetBrains"
        elif system == "Windows":
            appdata = os.environ.get("APPDATA")
            if appdata:
                return Path(appdata) / "JetBrains"
        return None

    def _scan_workspace_storage(self, storage_dir: Path) -> List[BootstrapMemory]:
        """Scan VS Code/Cursor workspaceStorage for state.vscdb with Copilot chat."""
        memories: List[BootstrapMemory] = []
        if not storage_dir.exists():
            return []

        for ws_dir in self._safe_iterdir(storage_dir):
            if not ws_dir.is_dir():
                continue

            project_name = self._resolve_workspace_name(ws_dir)
            db_path = ws_dir / "state.vscdb"
            if not db_path.exists():
                continue

            prompts = self._extract_from_sqlite(db_path)
            if prompts:
                content = f"Project: {project_name}\nCopilot chat topics:\n"
                content += "\n".join(f"- {p[:200]}" for p in prompts[:20])
                memories.append(BootstrapMemory(
                    name=f"Copilot project: {project_name}",
                    content=content[:self.max_chars_per_memory],
                    tags=["bootstrap", "copilot", project_name],
                    importance="context",
                    source_path=str(db_path),
                    source_type="copilot",
                ))

        return memories

    def _resolve_workspace_name(self, ws_dir: Path) -> str:
        """Try to extract project name from workspace.json."""
        ws_json = ws_dir / "workspace.json"
        if ws_json.exists():
            try:
                with open(ws_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    folder = data.get("folder", "")
                    if folder:
                        return Path(folder).name
            except (json.JSONDecodeError, OSError):
                pass
        return ws_dir.name

    def _extract_from_sqlite(self, db_path: Path) -> List[str]:
        """Safely extract Copilot chat prompts from a VS Code SQLite state DB."""
        prompts: List[str] = []
        tmp_dir = tempfile.mkdtemp()
        tmp_db = Path(tmp_dir) / "state_copy.db"

        try:
            shutil.copy2(db_path, tmp_db)
            conn = sqlite3.connect(str(tmp_db))
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT value FROM ItemTable "
                    "WHERE key LIKE '%chat%' OR key LIKE '%copilot%'"
                )
                for (value,) in cursor.fetchall():
                    if not value:
                        continue
                    try:
                        data = json.loads(value)
                        prompts.extend(self._extract_user_prompts(data))
                    except (json.JSONDecodeError, TypeError):
                        continue
            except sqlite3.OperationalError:
                pass
            finally:
                conn.close()
        except (OSError, sqlite3.Error):
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return prompts

    def _scan_global_storage(self, path: Path) -> List[BootstrapMemory]:
        """Scan a global storage directory for Copilot chat JSON files."""
        return self._scan_json_dir(path, "global")

    def _scan_json_dir(self, path: Path, label: str = "general") -> List[BootstrapMemory]:
        """Recursively scan a directory for JSON files with chat history."""
        memories: List[BootstrapMemory] = []
        if not path.exists():
            return []

        for root, _, files in os.walk(path):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                fpath = Path(root) / fname
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    prompts = self._extract_user_prompts(data)
                    if prompts:
                        content = "\n".join(f"- {p[:200]}" for p in prompts[:20])
                        memories.append(BootstrapMemory(
                            name=f"Copilot {label}: {fname}",
                            content=content[:self.max_chars_per_memory],
                            tags=["bootstrap", "copilot", label],
                            importance="context",
                            source_path=str(fpath),
                            source_type="copilot",
                        ))
                except (json.JSONDecodeError, OSError):
                    continue

        return memories

    def _extract_user_prompts(self, data: Any) -> List[str]:
        """Recursively extract user prompts from arbitrary JSON structures."""
        prompts: List[str] = []

        if isinstance(data, dict):
            role = data.get("role")
            text = data.get("text") or data.get("content") or data.get("value")
            if role == "user" and isinstance(text, str) and text.strip():
                prompts.append(text.strip())
            for val in data.values():
                prompts.extend(self._extract_user_prompts(val))

        elif isinstance(data, list):
            for item in data:
                prompts.extend(self._extract_user_prompts(item))

        return prompts

    @staticmethod
    def _safe_iterdir(path: Path) -> List[Path]:
        """List directory contents, returning empty list on error."""
        try:
            return list(path.iterdir())
        except (PermissionError, OSError):
            return []
