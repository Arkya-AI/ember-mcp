"""
One-time migration: ~/.anchor/ → ~/.ember/

Copies the Anchor data directory to the Ember location, renames the
anchor_id field to ember_id in all JSON files, and injects Shadow-Decay
default fields. The original ~/.anchor/ is preserved as a safety backup.
"""

import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path.home() / ".anchor"
EMBER_DIR = Path.home() / ".ember"

# Shadow-Decay fields to inject if missing
SHADOW_DEFAULTS = {
    "shadow_load": 0.0,
    "shadowed_by": None,
    "shadow_updated_at": None,
    "related_ids": [],
    "superseded_by_id": None,
}


def migrate_anchor_to_ember(force: bool = False) -> dict:
    """
    Migrate ~/.anchor/ → ~/.ember/ with field renaming and Shadow-Decay defaults.

    Args:
        force: If True, overwrite existing ~/.ember/ directory.

    Returns:
        dict with keys: migrated, skipped, errors, total
    """
    result = {"migrated": 0, "skipped": 0, "errors": 0, "total": 0}

    if not ANCHOR_DIR.exists():
        print(f"  ✗ Source directory {ANCHOR_DIR} not found. Nothing to migrate.")
        return result

    if EMBER_DIR.exists() and not force:
        print(f"  ✗ Target directory {EMBER_DIR} already exists. Use --force to overwrite.")
        return result

    # Step 1: Copy entire directory tree
    print(f"  Copying {ANCHOR_DIR} → {EMBER_DIR}...")
    if EMBER_DIR.exists():
        shutil.rmtree(EMBER_DIR)
    shutil.copytree(ANCHOR_DIR, EMBER_DIR)
    print("  ✓ Directory copied")

    # Step 2: Rename anchors/ → embers/
    anchors_dir = EMBER_DIR / "anchors"
    embers_dir = EMBER_DIR / "embers"
    if anchors_dir.exists():
        anchors_dir.rename(embers_dir)
        print("  ✓ Renamed anchors/ → embers/")
    elif embers_dir.exists():
        print("  ✓ embers/ directory already exists")
    else:
        embers_dir.mkdir(exist_ok=True)
        print("  ⚠ No anchors/ or embers/ found, created empty embers/")

    # Step 3: Transform each JSON file
    json_files = list(embers_dir.glob("*.json"))
    result["total"] = len(json_files)
    print(f"  Transforming {len(json_files)} ember files...")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Rename anchor_id → ember_id
            if "anchor_id" in data:
                data["ember_id"] = data.pop("anchor_id")

            # Inject Shadow-Decay defaults for missing fields
            for key, default in SHADOW_DEFAULTS.items():
                if key not in data:
                    data[key] = default

            # Write back
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            result["migrated"] += 1

        except Exception as e:
            logger.error(f"Failed to migrate {json_file.name}: {e}")
            print(f"  ✗ Error on {json_file.name}: {e}")
            result["errors"] += 1

    # Step 4: Update config.json
    config_file = EMBER_DIR / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            config["data_dir"] = str(EMBER_DIR)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print("  ✓ Updated config.json data_dir")
        except Exception as e:
            print(f"  ⚠ Could not update config.json: {e}")

    print(f"\n  Migration complete: {result['migrated']} migrated, "
          f"{result['skipped']} skipped, {result['errors']} errors "
          f"(out of {result['total']} files)")
    print(f"  Original data preserved at {ANCHOR_DIR}")

    return result
