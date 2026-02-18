import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

from .models import EmberConfig

# Lazy imports — only loaded when first needed, not at module import time
_np = None
_faiss = None


def _get_np():
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _f
        _faiss = _f
    return _faiss


class VectorEngine:
    """
    Manages embeddings, FAISS indices, and Voronoi cell assignment.
    Handles persistence of the vector index and frozen centroids.
    All heavy imports (numpy, faiss, sentence-transformers) are deferred
    until first use to ensure fast MCP server startup.
    """

    def __init__(self, config: EmberConfig):
        self.config = config
        self.dimension = config.dimension
        self.k_cells = config.k_cells

        # Paths
        self.data_dir = Path(config.data_dir).expanduser()
        self.cells_dir = self.data_dir / "cells"
        self.index_dir = self.data_dir / "index"
        self.centroids_path = self.cells_dir / "centroids.npy"
        self.index_path = self.index_dir / "vectors.faiss"

        # Ensure directories exist
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Lazy loading for model and indices
        self._model = None
        self._model_name = config.model_name
        self._centroid_index = None
        self._memory_index = None

    @property
    def centroid_index(self):
        if self._centroid_index is None:
            self._centroid_index = self._load_or_create_centroids()
        return self._centroid_index

    @property
    def memory_index(self):
        if self._memory_index is None:
            self._memory_index = self._load_or_create_memory_index()
        return self._memory_index

    @property
    def model(self):
        """Lazy loader for the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _load_or_create_centroids(self):
        """
        Load frozen centroids from disk or generate new random unit vectors.
        Returns a populated IndexFlatL2.
        """
        np = _get_np()
        faiss = _get_faiss()
        index = faiss.IndexFlatL2(self.dimension)

        if self.centroids_path.exists():
            centroids = np.load(self.centroids_path)
            if centroids.shape != (self.k_cells, self.dimension):
                raise ValueError(
                    f"Corrupt centroids file. Expected shape ({self.k_cells}, {self.dimension}), "
                    f"got {centroids.shape}"
                )
        else:
            # Generate random vectors with fixed seed for reproducibility
            rng = np.random.default_rng(seed=42)
            centroids = rng.standard_normal((self.k_cells, self.dimension)).astype("float32")

            # Normalize to unit vectors (L2)
            faiss.normalize_L2(centroids)

            # Save to disk (frozen — never modified after creation)
            np.save(self.centroids_path, centroids)

        index.add(centroids)
        return index

    def _load_or_create_memory_index(self):
        """
        Load the memory index from disk or create a new IndexIDMap.
        """
        faiss = _get_faiss()
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                return index
            except Exception as e:
                raise RuntimeError(f"Failed to load FAISS index from {self.index_path}: {e}")

        # Create new index: IDMap wraps FlatL2 to allow custom integer IDs
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIDMap(quantizer)
        return index

    def embed(self, text: str):
        """
        Encode text into a normalized vector.

        Returns:
            np.ndarray: Shape (1, dim) float32 array.
        """
        np = _get_np()
        faiss = _get_faiss()
        vector = self.model.encode(text)
        vector = vector.astype("float32")

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Normalize L2
        faiss.normalize_L2(vector)

        # Safety assertion: verify normalization succeeded
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-5, f"Vector normalization failed: norm={norm}"

        return vector

    def assign_cell(self, vector) -> int:
        """
        Determine which Voronoi cell a vector belongs to.

        Args:
            vector: Shape (1, dim) float32 array.

        Returns:
            int: The index of the nearest centroid (0 to k_cells-1).
        """
        _, I = self.centroid_index.search(vector, 1)
        return int(I[0][0])

    def add_vector(self, faiss_id: int, vector) -> None:
        """
        Add a vector to the memory index with a specific ID and persist changes.
        """
        np = _get_np()
        ids_array = np.array([faiss_id], dtype=np.int64)

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        self.memory_index.add_with_ids(vector, ids_array)
        self.save_index()

    def remove_vector(self, faiss_id: int) -> None:
        """
        Remove a vector from the memory index by ID and persist changes.
        """
        np = _get_np()
        ids_array = np.array([faiss_id], dtype=np.int64)
        self.memory_index.remove_ids(ids_array)
        self.save_index()

    def search(self, query_vector, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search the memory index for nearest neighbors.

        Returns:
            List of (faiss_id, l2_squared_distance) tuples.
        """
        if self.memory_index.ntotal == 0:
            return []

        k = min(top_k, self.memory_index.ntotal)
        distances, indices = self.memory_index.search(query_vector, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))

        return results

    def search_radius(self, query_vector, radius_l2: float) -> List[Tuple[int, float]]:
        """
        Find all vectors within a given L2 radius using FAISS range_search.

        Args:
            query_vector: Shape (1, dim) float32 array.
            radius_l2: L2 distance threshold (not squared).

        Returns:
            List of (faiss_id, l2_squared_distance) tuples.
        """
        if self.memory_index.ntotal == 0:
            return []

        # FAISS range_search expects squared L2 distance as threshold
        threshold = radius_l2 * radius_l2
        lims, D, I = self.memory_index.range_search(query_vector, threshold)

        results = []
        # query_vector is 1xDim, so results are between lims[0] and lims[1]
        start = int(lims[0])
        end = int(lims[1])

        for i in range(start, end):
            results.append((int(I[i]), float(D[i])))

        return results

    def reconstruct_vector(self, faiss_id: int) -> Optional["np.ndarray"]:
        """
        Reconstruct a vector from the FAISS index by its integer ID.

        Args:
            faiss_id: The integer FAISS ID.

        Returns:
            np.ndarray of shape (dimension,) float32, or None if ID not found.
        """
        try:
            return self.memory_index.reconstruct(faiss_id)
        except RuntimeError:
            return None

    def save_index(self) -> None:
        """Persist the memory index to disk."""
        faiss = _get_faiss()
        faiss.write_index(self.memory_index, str(self.index_path))

    def reload_index(self) -> None:
        """Reload the memory index from disk. Picks up writes from other processes."""
        self._memory_index = self._load_or_create_memory_index()

    def get_centroids(self):
        """
        Return the raw centroid array, shape (k_cells, dimension).
        Loads from disk if needed (triggers centroid_index init).
        """
        np = _get_np()
        if self.centroids_path.exists():
            return np.load(self.centroids_path)
        # Trigger creation + load
        _ = self.centroid_index
        return np.load(self.centroids_path)

    def get_cell_for_vectors(self, faiss_ids: List[int]) -> Dict[int, int]:
        """
        Reconstruct vectors from IDs and determine their cell assignments.

        Returns:
            Dictionary mapping {faiss_id: cell_id}.
        """
        result = {}

        for fid in faiss_ids:
            vec = self.reconstruct_vector(fid)
            if vec is not None:
                vec = vec.reshape(1, -1)
                cell_id = self.assign_cell(vec)
                result[fid] = cell_id

        return result

    @staticmethod
    def l2_to_cosine(l2_sq: float) -> float:
        """
        Convert FAISS L2 squared distance to cosine similarity.

        For L2-normalized vectors: L2² = 2(1 - cos_sim)
        Therefore: cos_sim = 1 - (L2² / 2)

        FAISS IndexFlatL2 returns L2² (squared distance).

        Args:
            l2_sq: Squared L2 distance from FAISS.

        Returns:
            Cosine similarity clamped to [0.0, 1.0].
        """
        cos_sim = 1.0 - (l2_sq / 2.0)
        return max(0.0, min(1.0, cos_sim))

    @staticmethod
    def cosine_to_l2(cos_sim: float) -> float:
        """
        Convert cosine similarity to L2 distance for normalized vectors.

        Reverse of l2_to_cosine: L2 = sqrt(2 * (1 - cos_sim))

        Args:
            cos_sim: Cosine similarity value.

        Returns:
            L2 distance (not squared).
        """
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return math.sqrt(2.0 * (1.0 - cos_sim))
