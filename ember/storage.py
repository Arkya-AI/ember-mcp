import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Set

import aiofiles
import aiosqlite
from filelock import FileLock

from ember.models import Ember, EmberConfig, RegionStats

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Persistence layer for the Ember MCP server.

    Handles:
    - JSON file storage for Ember documents.
    - SQLite storage for edges, region stats, and metrics.
    - ID mapping between UUIDs (Embers) and Integers (FAISS).
    """

    def __init__(self, config: EmberConfig):
        self.config = config
        self.data_dir = Path(config.data_dir).expanduser()
        self.embers_dir = self.data_dir / "embers"
        self.index_dir = self.data_dir / "index"
        self.db_path = self.data_dir / "cells" / "stats.db"
        self.id_map_path = self.index_dir / "id_map.json"

        # In-memory ID maps
        self.int_to_uuid: Dict[int, str] = {}
        self.uuid_to_int: Dict[str, int] = {}
        self.next_int_id: int = 0

        # Concurrency locks
        self._map_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()

        # Cross-process file lock for id_map.json
        self._id_map_file_lock = FileLock(str(self.id_map_path) + ".lock")

    async def init_db(self):
        """Initialize storage directories, SQLite database, and load ID maps."""
        try:
            self.embers_dir.mkdir(parents=True, exist_ok=True)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            async with self._db_lock:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS edges (
                            source_id TEXT NOT NULL,
                            target_id TEXT NOT NULL,
                            edge_type TEXT NOT NULL,
                            weight REAL DEFAULT 0.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (source_id, target_id, edge_type)
                        )
                    """)
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS region_stats (
                            cell_id INTEGER PRIMARY KEY,
                            vitality_score REAL DEFAULT 0.0,
                            shadow_accum REAL DEFAULT 0.0,
                            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS metrics_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metric_type TEXT NOT NULL,
                            value REAL,
                            details TEXT
                        )
                    """)
                    await db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
                    )
                    await db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
                    )
                    await db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics_log(metric_type)"
                    )
                    await db.commit()

            await self._load_id_map()
            logger.info(f"Storage initialized at {self.data_dir}")

        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise

    # ------------------------------------------------------------------
    # ID Map Management
    # ------------------------------------------------------------------

    async def _load_id_map(self):
        """Load the UUID <-> Int ID mapping from disk."""
        async with self._map_lock:
            if not self.id_map_path.exists():
                self.int_to_uuid = {}
                self.uuid_to_int = {}
                self.next_int_id = 0
                return

            try:
                async with aiofiles.open(self.id_map_path, mode="r") as f:
                    content = await f.read()
                    if not content:
                        data = {}
                    else:
                        data = json.loads(content)

                self.int_to_uuid = {int(k): v for k, v in data.items()}
                self.uuid_to_int = {v: k for k, v in self.int_to_uuid.items()}

                if self.int_to_uuid:
                    self.next_int_id = max(self.int_to_uuid.keys()) + 1
                else:
                    self.next_int_id = 0

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Corrupted ID map file: {e}. Backing up and resetting.")
                if self.id_map_path.exists():
                    os.rename(self.id_map_path, self.id_map_path.with_suffix(".json.bak"))
                self.int_to_uuid = {}
                self.uuid_to_int = {}
                self.next_int_id = 0

    async def _save_id_map(self):
        """Persist the ID mapping to disk with cross-process locking."""
        try:
            data = {str(k): v for k, v in self.int_to_uuid.items()}
            with self._id_map_file_lock:
                async with aiofiles.open(self.id_map_path, mode="w") as f:
                    await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save ID map: {e}")
            raise

    async def reload_id_map(self):
        """Reload the UUID <-> Int ID mapping from disk. Picks up writes from other processes."""
        await self._load_id_map()

    # ------------------------------------------------------------------
    # Ember CRUD
    # ------------------------------------------------------------------

    async def save_ember(self, ember: Ember) -> int:
        """
        Save an Ember to JSON and assign/retrieve its integer FAISS ID.

        Returns:
            The integer FAISS ID for this ember.
        """
        ember_id = ember.ember_id
        file_path = self.embers_dir / f"{ember_id}.json"

        # Write JSON file
        try:
            json_str = ember.model_dump_json(indent=2)
            async with aiofiles.open(file_path, mode="w") as f:
                await f.write(json_str)
        except Exception as e:
            logger.error(f"Failed to write ember file {file_path}: {e}")
            raise

        # Update ID Map
        async with self._map_lock:
            if ember_id in self.uuid_to_int:
                return self.uuid_to_int[ember_id]
            else:
                int_id = self.next_int_id
                self.next_int_id += 1

                self.int_to_uuid[int_id] = ember_id
                self.uuid_to_int[ember_id] = int_id

                await self._save_id_map()
                return int_id

    async def get_ember(self, ember_id: str) -> Optional[Ember]:
        """Retrieve an Ember by its UUID."""
        file_path = self.embers_dir / f"{ember_id}.json"

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, mode="r") as f:
                content = await f.read()
                return Ember.model_validate_json(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to decode ember {ember_id}: {e}")
            return None

    async def delete_ember(self, ember_id: str) -> Optional[int]:
        """
        Delete an ember file and remove it from the ID map.
        Returns the integer FAISS ID if found, else None.
        """
        file_path = self.embers_dir / f"{ember_id}.json"

        try:
            if file_path.exists():
                os.remove(file_path)
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e}")

        async with self._map_lock:
            if ember_id not in self.uuid_to_int:
                return None

            int_id = self.uuid_to_int[ember_id]

            del self.uuid_to_int[ember_id]
            if int_id in self.int_to_uuid:
                del self.int_to_uuid[int_id]

            await self._save_id_map()
            return int_id

    async def update_ember(self, ember: Ember) -> None:
        """
        Update an existing ember on disk. Used for updating access stats,
        staleness flags, and other mutable fields.
        """
        file_path = self.embers_dir / f"{ember.ember_id}.json"

        if not file_path.exists():
            logger.warning(f"Cannot update non-existent ember: {ember.ember_id}")
            return

        try:
            ember.updated_at = datetime.now(timezone.utc)
            json_str = ember.model_dump_json(indent=2)
            async with aiofiles.open(file_path, mode="w") as f:
                await f.write(json_str)
        except Exception as e:
            logger.error(f"Failed to update ember {ember.ember_id}: {e}")
            raise

    async def list_embers(self, tag: Optional[str] = None) -> List[Ember]:
        """List all embers, optionally filtering by a specific tag."""
        embers: List[Ember] = []

        try:
            paths = list(self.embers_dir.glob("*.json"))

            for path in paths:
                try:
                    async with aiofiles.open(path, mode="r") as f:
                        content = await f.read()
                        ember = Ember.model_validate_json(content)

                        if tag:
                            if tag in ember.tags:
                                embers.append(ember)
                        else:
                            embers.append(ember)
                except Exception as e:
                    logger.warning(f"Skipping corrupted ember {path.name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error listing embers: {e}")
            return []

        return embers

    # ------------------------------------------------------------------
    # Knowledge Graph: Edges
    # ------------------------------------------------------------------

    async def save_edge(
        self, source_id: str, target_id: str, edge_type: str, weight: float = 0.0
    ) -> None:
        """Upsert an edge in the knowledge graph."""
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, weight, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (source_id, target_id, edge_type, weight, datetime.now(timezone.utc).isoformat()),
                )
                await db.commit()

    async def get_edges(
        self, ember_id: str, edge_type: Optional[str] = None
    ) -> List[Dict]:
        """Get all edges involving an ember (as source or target)."""
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                if edge_type:
                    cursor = await db.execute(
                        """
                        SELECT source_id, target_id, edge_type, weight, created_at
                        FROM edges
                        WHERE (source_id = ? OR target_id = ?) AND edge_type = ?
                        """,
                        (ember_id, ember_id, edge_type),
                    )
                else:
                    cursor = await db.execute(
                        """
                        SELECT source_id, target_id, edge_type, weight, created_at
                        FROM edges
                        WHERE source_id = ? OR target_id = ?
                        """,
                        (ember_id, ember_id),
                    )
                rows = await cursor.fetchall()

        return [
            {
                "source_id": row[0],
                "target_id": row[1],
                "edge_type": row[2],
                "weight": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    async def get_neighbors(
        self, ember_id: str, edge_type: Optional[str] = None
    ) -> List[str]:
        """Get connected ember IDs (both directions), deduplicated."""
        edges = await self.get_edges(ember_id, edge_type)
        neighbors = set()
        for edge in edges:
            if edge["source_id"] != ember_id:
                neighbors.add(edge["source_id"])
            if edge["target_id"] != ember_id:
                neighbors.add(edge["target_id"])
        return list(neighbors)

    async def traverse_kg(
        self,
        start_id: str,
        depth: int = 2,
        edge_types: Optional[List[str]] = None,
    ) -> Set[str]:
        """
        BFS traversal from start_id up to `depth` hops.

        Returns set of ember_ids found (excluding start_id).
        """
        visited: Set[str] = set()
        queue: deque = deque()
        queue.append((start_id, 0))
        visited.add(start_id)

        while queue:
            current_id, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Get neighbors for each edge type or all types
            if edge_types:
                neighbors: Set[str] = set()
                for et in edge_types:
                    et_neighbors = await self.get_neighbors(current_id, et)
                    neighbors.update(et_neighbors)
            else:
                neighbors = set(await self.get_neighbors(current_id))

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, current_depth + 1))

        # Exclude start_id from results
        visited.discard(start_id)
        return visited

    # ------------------------------------------------------------------
    # Region Stats (Conflict Density)
    # ------------------------------------------------------------------

    async def update_region(
        self, cell_id: int, vitality: float, shadow_accum: float
    ) -> None:
        """Upsert per-cell region statistics."""
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO region_stats (cell_id, vitality_score, shadow_accum, last_updated)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cell_id, vitality, shadow_accum, datetime.now(timezone.utc).isoformat()),
                )
                await db.commit()

    async def get_region_stats(self, cell_id: int) -> Optional[RegionStats]:
        """Retrieve region stats for a specific Voronoi cell."""
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT cell_id, vitality_score, shadow_accum, last_updated FROM region_stats WHERE cell_id = ?",
                    (cell_id,),
                )
                row = await cursor.fetchone()

        if not row:
            return None

        return RegionStats(
            cell_id=row[0],
            vitality_score=row[1],
            shadow_accum=row[2],
            last_updated=datetime.fromisoformat(row[3]) if row[3] else datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Metrics Log
    # ------------------------------------------------------------------

    async def log_metric(
        self, metric_type: str, value: float, details_dict: Optional[dict] = None
    ) -> None:
        """Append a metric entry to the metrics log."""
        details_json = json.dumps(details_dict) if details_dict else None

        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO metrics_log (timestamp, metric_type, value, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (datetime.now(timezone.utc).isoformat(), metric_type, value, details_json),
                )
                await db.commit()

    async def get_metric_history(
        self, metric_type: str, limit: int = 10
    ) -> List[Dict]:
        """Get recent metric values, ordered by newest first."""
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT id, timestamp, metric_type, value, details
                    FROM metrics_log
                    WHERE metric_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (metric_type, limit),
                )
                rows = await cursor.fetchall()

        results = []
        for row in rows:
            details = None
            if row[4]:
                try:
                    details = json.loads(row[4])
                except (json.JSONDecodeError, TypeError):
                    details = row[4]

            results.append({
                "id": row[0],
                "timestamp": row[1],
                "metric_type": row[2],
                "value": row[3],
                "details": details,
            })

        return results
