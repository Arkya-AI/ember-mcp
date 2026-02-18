"""Cold-start bootstrap scanner — pre-populates memories from user's machine."""

import asyncio
import json
import os
import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from ember.models import Ember


@dataclass
class BootstrapMemory:
    """Intermediate DTO for a potential memory from scanning."""
    name: str
    content: str
    tags: List[str] = field(default_factory=list)
    importance: str = "context"
    source_path: str = ""
    source_type: str = ""  # claude-code, git, document


class BaseScanner(ABC):
    """Abstract base class for data source scanners."""

    @abstractmethod
    async def scan(self) -> List[BootstrapMemory]:
        """Scan a data source and return potential memories."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for progress output."""
        pass


# Filename patterns to always skip (security)
SKIP_FILENAMES = {".env", ".pem", "id_rsa", "id_ed25519", "credentials",
                  "secrets.yaml", "secrets.json", ".npmrc", ".pypirc"}

SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv",
             ".tox", ".mypy_cache", ".pytest_cache", "dist", "build"}


class DocumentScanner(BaseScanner):
    """Scans ~/Desktop and ~/Documents for .md and .txt files."""

    def __init__(self, max_depth: int = 2, max_file_size: int = 10240,
                 max_words: int = 500):
        self.scan_dirs = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
        ]
        self.max_depth = max_depth
        self.max_file_size = max_file_size
        self.max_words = max_words
        self.extensions = {".md", ".txt"}

    @property
    def display_name(self) -> str:
        return "documents"

    async def scan(self) -> List[BootstrapMemory]:
        memories: List[BootstrapMemory] = []
        for base_dir in self.scan_dirs:
            if not base_dir.exists():
                continue
            self._scan_dir(base_dir, 0, memories)
        return memories

    def _scan_dir(self, directory: Path, depth: int,
                  memories: List[BootstrapMemory]) -> None:
        if depth > self.max_depth:
            return
        try:
            for entry in directory.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    if entry.name in SKIP_DIRS:
                        continue
                    self._scan_dir(entry, depth + 1, memories)
                elif entry.is_file():
                    if entry.name.lower() in SKIP_FILENAMES:
                        continue
                    if entry.suffix.lower() in self.extensions:
                        mem = self._process_file(entry)
                        if mem:
                            memories.append(mem)
        except PermissionError:
            pass

    def _process_file(self, path: Path) -> Optional[BootstrapMemory]:
        try:
            if path.stat().st_size > self.max_file_size:
                return None
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return None
            words = text.split()
            truncated = " ".join(words[:self.max_words])
            return BootstrapMemory(
                name=f"Document: {path.name}",
                content=truncated,
                tags=["bootstrap", "document"],
                importance="context",
                source_path=str(path),
                source_type="document",
            )
        except (OSError, UnicodeDecodeError):
            return None


TECH_STACK_FILES = {
    "package.json": "javascript",
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "Gemfile": "ruby",
    "pom.xml": "java",
    "build.gradle": "java",
    "composer.json": "php",
    "mix.exs": "elixir",
}


class GitRepoScanner(BaseScanner):
    """Scans git repos with recent activity for READMEs, tech stacks, and commit history."""

    def __init__(self, max_days: int = 90, max_repos: int = 30, max_depth: int = 3):
        self.scan_dirs = [
            Path.home() / "Documents",
            Path.home() / "Projects",
            Path.home() / "Developer",
            Path.home() / "Desktop",
        ]
        self.max_days = max_days
        self.max_repos = max_repos
        self.max_depth = max_depth

    @property
    def display_name(self) -> str:
        return "git repositories"

    async def scan(self) -> List[BootstrapMemory]:
        repos = self._find_repos()
        memories: List[BootstrapMemory] = []
        for repo_path in repos[:self.max_repos]:
            repo_name = repo_path.name
            readme_mem = self._read_readme(repo_path, repo_name)
            if readme_mem:
                memories.append(readme_mem)
            stack_mem = self._detect_tech_stack(repo_path, repo_name)
            if stack_mem:
                memories.append(stack_mem)
            commits_mem = self._read_commits(repo_path, repo_name)
            if commits_mem:
                memories.append(commits_mem)
        return memories

    def _find_repos(self) -> List[Path]:
        repos = []
        cutoff = datetime.now() - timedelta(days=self.max_days)
        for base in self.scan_dirs:
            if not base.exists():
                continue
            try:
                for git_dir in base.rglob(".git"):
                    if any(part in SKIP_DIRS for part in git_dir.parts):
                        continue
                    rel = git_dir.relative_to(base)
                    if len(rel.parts) > self.max_depth + 1:
                        continue
                    repo = git_dir.parent
                    if self._has_recent_commits(repo, cutoff):
                        repos.append(repo)
            except (PermissionError, OSError):
                continue
        return repos

    def _has_recent_commits(self, repo: Path, cutoff: datetime) -> bool:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%aI"],
                cwd=repo, capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False
            last_date = datetime.fromisoformat(result.stdout.strip().replace("Z", "+00:00"))
            return last_date.replace(tzinfo=None) > cutoff
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return False

    def _read_readme(self, repo: Path, name: str) -> Optional[BootstrapMemory]:
        for readme_name in ["README.md", "readme.md", "README.txt", "README"]:
            readme = repo / readme_name
            if readme.exists():
                try:
                    text = readme.read_text(encoding="utf-8", errors="ignore")
                    words = text.split()[:500]
                    if not words:
                        continue
                    return BootstrapMemory(
                        name=f"Project: {name}",
                        content=" ".join(words),
                        tags=["bootstrap", "git", name],
                        importance="context",
                        source_path=str(readme),
                        source_type="git",
                    )
                except OSError:
                    continue
        return None

    def _detect_tech_stack(self, repo: Path, name: str) -> Optional[BootstrapMemory]:
        detected = []
        details = []
        for filename, lang in TECH_STACK_FILES.items():
            fpath = repo / filename
            if fpath.exists():
                detected.append(lang)
                if filename == "package.json":
                    try:
                        data = json.loads(fpath.read_text(encoding="utf-8"))
                        deps = list(data.get("dependencies", {}).keys())[:10]
                        dev_deps = list(data.get("devDependencies", {}).keys())[:5]
                        if deps:
                            details.append(f"Dependencies: {', '.join(deps)}")
                        if dev_deps:
                            details.append(f"Dev dependencies: {', '.join(dev_deps)}")
                    except (json.JSONDecodeError, OSError):
                        pass
                elif filename == "pyproject.toml":
                    try:
                        text = fpath.read_text(encoding="utf-8")
                        if "dependencies" in text:
                            details.append(f"Python project config found")
                    except OSError:
                        pass
        if not detected:
            return None
        content = f"Project '{name}' uses: {', '.join(set(detected))}."
        if details:
            content += " " + ". ".join(details)
        return BootstrapMemory(
            name=f"Tech stack: {name}",
            content=content,
            tags=["bootstrap", "git", name, "tech-stack"],
            importance="fact",
            source_path=str(repo),
            source_type="git",
        )

    def _read_commits(self, repo: Path, name: str) -> Optional[BootstrapMemory]:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-20"],
                cwd=repo, capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            commits = result.stdout.strip()
            return BootstrapMemory(
                name=f"Recent activity: {name}",
                content=f"Last 20 commits in {name}:\n{commits}",
                tags=["bootstrap", "git", name, "activity"],
                importance="context",
                source_path=str(repo),
                source_type="git",
            )
        except (subprocess.TimeoutExpired, OSError):
            return None


class BootstrapPipeline:
    """Orchestrates scanning, deduplication, and storage of bootstrap memories."""

    def __init__(self, engine, storage):
        self.engine = engine
        self.storage = storage
        self.scanners: List[BaseScanner] = []
        self.similarity_threshold = 0.85

    def register_scanner(self, scanner: BaseScanner) -> None:
        self.scanners.append(scanner)

    def register_defaults(self) -> None:
        """Register all default scanners."""
        from ember.scanners.claude_code import ClaudeCodeScanner
        from ember.scanners.codex import CodexScanner
        from ember.scanners.copilot import CopilotScanner
        self.register_scanner(ClaudeCodeScanner())
        self.register_scanner(CodexScanner())
        self.register_scanner(CopilotScanner())
        self.register_scanner(GitRepoScanner())
        self.register_scanner(DocumentScanner())

    async def run(self, dry_run: bool = False) -> dict:
        print("\nBootstrapping memories from your machine...\n")
        all_memories: List[BootstrapMemory] = []
        scanner_stats = {}

        for scanner in self.scanners:
            print(f"  Scanning {scanner.display_name}...")
            try:
                memories = await scanner.scan()
                scanner_stats[scanner.display_name] = len(memories)
                all_memories.extend(memories)
                print(f"    ✓ Found {len(memories)} memories")
            except Exception as e:
                scanner_stats[scanner.display_name] = 0
                print(f"    ✗ Error: {e}")

        if not all_memories:
            print("\n  No memories found to bootstrap.")
            return {"total_scanned": 0, "total_stored": 0, "scanners": scanner_stats}

        print(f"\n  Deduplicating...")
        unique = self._deduplicate(all_memories)
        removed = len(all_memories) - len(unique)
        print(f"    ✓ {len(all_memories)} → {len(unique)} unique memories ({removed} duplicates removed)")

        if dry_run:
            print(f"\n  [Dry run] Would store {len(unique)} memories.")
            return {
                "total_scanned": len(all_memories),
                "total_stored": 0,
                "duplicates_removed": removed,
                "scanners": scanner_stats,
                "dry_run": True,
            }

        print(f"\n  Storing {len(unique)} memories...")
        stored = 0
        for mem in unique:
            try:
                ember = Ember(
                    name=mem.name,
                    content=mem.content,
                    tags=mem.tags,
                    importance=mem.importance,
                    source="bootstrap",
                    source_path=mem.source_path,
                )
                embedding = self.engine.embed(ember.content)
                cell_id = self.engine.assign_cell(embedding)
                ember.cell_id = cell_id
                int_id = await self.storage.save_ember(ember)
                self.engine.add_vector(int_id, embedding)
                # Shadow-on-Insert will be handled at the server level
                stored += 1
            except Exception as e:
                print(f"    ✗ Failed to store '{mem.name}': {e}")

        self.engine.save_index()
        print(f"    ✓ Stored {stored} memories")
        self._print_summary(scanner_stats, stored)

        return {
            "total_scanned": len(all_memories),
            "total_stored": stored,
            "duplicates_removed": removed,
            "scanners": scanner_stats,
        }

    def _deduplicate(self, memories: List[BootstrapMemory]) -> List[BootstrapMemory]:
        if len(memories) <= 1:
            return memories

        try:
            import numpy as np
            texts = [m.content for m in memories]
            embeddings = np.array([self.engine.embed(t).flatten() for t in texts])

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = embeddings / norms

            keep = [True] * len(memories)
            for i in range(len(memories)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(memories)):
                    if not keep[j]:
                        continue
                    sim = float(np.dot(normalized[i], normalized[j]))
                    if sim > self.similarity_threshold:
                        if len(memories[j].content) > len(memories[i].content):
                            keep[i] = False
                            memories[j].tags = list(set(memories[j].tags + memories[i].tags))
                            break
                        else:
                            keep[j] = False
                            memories[i].tags = list(set(memories[i].tags + memories[j].tags))

            return [m for m, k in zip(memories, keep) if k]
        except Exception:
            return memories

    def _print_summary(self, scanner_stats: dict, total_stored: int) -> None:
        print(f"\n  Bootstrap complete! Added {total_stored} memories.")
        print(f"  Your AI will feel like it's known you for months. ✨\n")
