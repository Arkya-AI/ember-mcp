import argparse
import asyncio
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EMBER_DIR = Path.home() / ".ember"
MCP_SERVER_ENTRY = {
    "command": "ember-mcp",
    "args": ["run"],
}


def _claude_desktop_path() -> Optional[Path]:
    system = platform.system().lower()
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "linux":
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    elif system == "windows":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return None


def _client_paths() -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    cd = _claude_desktop_path()
    if cd:
        paths["Claude Desktop"] = cd
    paths["Claude Code"] = Path.home() / ".claude.json"
    paths["Cursor"] = Path.home() / ".cursor" / "mcp.json"
    paths["Windsurf"] = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    return paths


def _register_client(name: str, path: Path) -> bool:
    """Add the ember entry to an MCP client config file."""
    if not path.exists():
        if not path.parent.exists():
            return False
        config: dict = {"mcpServers": {}}
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  ! Could not parse {path}, skipping.")
            return False

    # Backup before modifying
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.backup"))

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["ember"] = MCP_SERVER_ENTRY

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except OSError as exc:
        print(f"  ! Write failed for {path}: {exc}")
        return False


def _init_storage() -> None:
    print("\nInitializing storage at ~/.ember/...")
    try:
        EMBER_DIR.mkdir(parents=True, exist_ok=True)
        (EMBER_DIR / "embers").mkdir(exist_ok=True)
        (EMBER_DIR / "index").mkdir(exist_ok=True)
        (EMBER_DIR / "cells").mkdir(exist_ok=True)
        print("  \u2713 Created directories")
    except OSError as exc:
        print(f"  \u2717 Failed: {exc}")


def _download_model() -> None:
    print("\nDownloading embedding model (first time only)...")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        print("  \u2713 Model ready")
    except ImportError:
        print("  \u2717 sentence-transformers not installed. Run: pip install ember-mcp")
    except Exception as exc:
        print(f"  \u2717 Download failed: {exc}")


# ── Subcommands ──────────────────────────────────────────────────────

def cmd_init(_args: argparse.Namespace) -> None:
    print("Ember MCP \u2014 Setup\n")

    print("Checking for MCP clients...")
    all_clients = _client_paths()
    detected: List[Tuple[str, Path]] = []

    for name, path in all_clients.items():
        if path.exists() or path.parent.exists():
            print(f"  \u2713 {name} found")
            detected.append((name, path))
        else:
            print(f"  \u2717 {name} not found")

    if detected:
        print("\nRegistering Ember with detected clients...")
        for name, path in detected:
            if _register_client(name, path):
                print(f"  \u2713 {name} \u2014 registered")
            else:
                print(f"  \u2717 {name} \u2014 failed")
    else:
        print("\nNo supported MCP clients detected.")
        print("You can manually add Ember to your client's MCP config:")
        print(f'  {json.dumps({"mcpServers": {"ember": MCP_SERVER_ENTRY}}, indent=2)}')

    _init_storage()
    _download_model()

    # Run bootstrap to pre-populate memories
    print("\n" + "\u2500" * 50)
    print("Running memory bootstrap...")
    try:
        bootstrap_args = argparse.Namespace(sources=None, dry_run=False)
        cmd_bootstrap(bootstrap_args)
    except Exception as e:
        print(f"  Bootstrap encountered an error: {e}")
        print("  You can re-run with: ember-mcp bootstrap")

    print("\n" + "\u2500" * 50)
    print("Setup complete! Restart your AI clients to activate Ember.")
    print("Your AI now has persistent memory across all sessions.")


def cmd_status(_args: argparse.Namespace) -> None:
    print("Ember MCP \u2014 Status\n")

    if EMBER_DIR.exists():
        ember_count = len(list((EMBER_DIR / "embers").glob("*.json"))) if (EMBER_DIR / "embers").exists() else 0
        print(f"  \u2713 Storage: {EMBER_DIR} ({ember_count} memories)")
    else:
        print("  \u2717 Storage: not initialized (run 'ember-mcp init')")

    print("\nClient registrations:")
    for name, path in _client_paths().items():
        configured = False
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                configured = "ember" in data.get("mcpServers", {})
            except (json.JSONDecodeError, OSError):
                pass
        mark = "\u2713" if configured else "\u2013"
        label = "registered" if configured else "not registered"
        print(f"  {mark} {name}: {label}")


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Run the bootstrap scanner to pre-populate memories."""
    print("Ember MCP — Bootstrap\n")

    from ember.config import load_config
    from ember.core import VectorEngine
    from ember.storage import StorageManager
    from ember.bootstrap import (
        BootstrapPipeline, GitRepoScanner, DocumentScanner,
    )
    from ember.scanners.claude_code import ClaudeCodeScanner
    from ember.scanners.codex import CodexScanner
    from ember.scanners.copilot import CopilotScanner

    config = load_config()
    engine = VectorEngine(config)
    storage = StorageManager(config)

    async def _run():
        await storage.init_db()
        pipeline = BootstrapPipeline(engine, storage)

        sources = getattr(args, "sources", None)
        if sources:
            source_list = [s.strip().lower() for s in sources.split(",")]
            if "claude" in source_list:
                pipeline.register_scanner(ClaudeCodeScanner())
            if "codex" in source_list:
                pipeline.register_scanner(CodexScanner())
            if "copilot" in source_list:
                pipeline.register_scanner(CopilotScanner())
            if "git" in source_list:
                pipeline.register_scanner(GitRepoScanner())
            if "doc" in source_list or "document" in source_list:
                pipeline.register_scanner(DocumentScanner())
        else:
            pipeline.register_defaults()

        dry_run = getattr(args, "dry_run", False)
        await pipeline.run(dry_run=dry_run)

    asyncio.run(_run())


def cmd_migrate(args: argparse.Namespace) -> None:
    """Migrate data from ~/.anchor/ to ~/.ember/."""
    print("Ember MCP — Migration\n")
    from ember.migration import migrate_anchor_to_ember
    force = getattr(args, "force", False)
    result = migrate_anchor_to_ember(force=force)
    if result["migrated"] > 0:
        print("\n  ✓ Migration successful. Restart your AI clients to use Ember.")
    elif result["total"] == 0:
        print("\n  Nothing to migrate.")


def cmd_run(_args: argparse.Namespace) -> None:
    from ember.server import mcp
    mcp.run()


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ember-mcp",
        description="Ember MCP \u2014 Persistent memory for AI with drift detection",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="Set up Ember and register with all detected MCP clients")
    sub.add_parser("run", help="Start the Ember MCP server")
    sub.add_parser("status", help="Show configuration and registration status")
    migrate_parser = sub.add_parser("migrate", help="Migrate data from ~/.anchor/ to ~/.ember/")
    migrate_parser.add_argument("--force", action="store_true",
        help="Overwrite existing ~/.ember/ directory")
    bootstrap_parser = sub.add_parser("bootstrap", help="Scan your machine and pre-populate AI memories")
    bootstrap_parser.add_argument("--sources", type=str, default=None,
        help="Comma-separated scanners: claude,codex,copilot,git,doc (default: all)")
    bootstrap_parser.add_argument("--dry-run", action="store_true",
        help="Show what would be scanned without storing")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "bootstrap":
        cmd_bootstrap(args)
    else:
        # No subcommand: auto-run init (one-command install experience)
        cmd_init(args)


if __name__ == "__main__":
    main()
