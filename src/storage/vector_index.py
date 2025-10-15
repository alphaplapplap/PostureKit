"""
Vector index management for pose similarity search.
Integrates FAISS with PostgreSQL for synchronized vector storage.
"""

import faiss  # type: ignore
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import pickle
import logging
from uuid import UUID

from src.utils.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class SimilarPose:
    """
    A similar pose found through vector search.

    Attributes:
        pose_id: UUID of the similar pose
        distance: L2 distance in feature space (lower = more similar)
        similarity_score: Normalized similarity (0-1, higher = more similar)
    """

    pose_id: UUID
    distance: float
    similarity_score: float

    def __post_init__(self) -> None:
        """Validate similarity metrics."""
        assert (
            self.distance >= 0.0
        ), f"Distance must be non-negative, got {self.distance}"
        assert (
            0.0 <= self.similarity_score <= 1.0
        ), f"Similarity must be in [0,1], got {self.similarity_score}"


class VectorIndexError(Exception):
    """Base exception for vector index errors."""

    pass


class VectorIndex:
    """
    FAISS-based vector index for pose similarity search.

    This class maintains a FAISS index synchronized with PostgreSQL.
    Every add/remove operation must be paired with corresponding database operations
    to prevent desynchronization.

    The index uses L2 (Euclidean) distance as the similarity metric, which is
    appropriate for normalized feature vectors. Lower distances indicate higher similarity.

    Attributes:
        dimension: Feature vector dimensionality (GEOMETRIC_FEATURE_DIM for geometric features)
        index_path: Path to persisted FAISS index file
        index: FAISS IndexFlatL2 for exact nearest neighbor search
        id_map: Maps FAISS internal indices to pose UUIDs
    """

    def __init__(self, dimension: Optional[int] = None, index_path: Optional[Path] = None):
        """
        Initialize vector index.

        Args:
            dimension: Feature vector dimension (must match extracted features).
                      Defaults to settings.GEOMETRIC_FEATURE_DIM if not provided.
            index_path: Path to load/save index. If None, index is memory-only.
        """
        # Use default dimension from settings if not provided
        self.dimension = dimension if dimension is not None else settings.GEOMETRIC_FEATURE_DIM
        self.index_path = index_path

        # Create FAISS index using L2 (Euclidean) distance
        # IndexFlatL2 provides exact search with no approximation
        # For larger datasets (>1M vectors), consider IndexIVFFlat for speed
        self.index = faiss.IndexFlatL2(self.dimension)

        # Map FAISS internal indices (0, 1, 2...) to pose UUIDs
        # This is necessary because FAISS uses sequential integer IDs
        self.id_map: Dict[int, UUID] = {}

        # Load existing index if path provided and file exists
        if index_path and index_path.exists():
            self.load()

        logger.info(
            f"VectorIndex initialized: dimension={self.dimension}, size={self.size()}",
            extra={
                "extra_data": {
                    "dimension": self.dimension,
                    "index_path": str(index_path) if index_path else None,
                }
            },
        )

    def add(self, pose_id: UUID, feature_vector: np.ndarray) -> None:
        """
        Add feature vector to index.

        This should always be called within the same transaction as the
        PostgreSQL insert to maintain synchronization.

        Args:
            pose_id: UUID of the pose
            feature_vector: Feature vector (dimension,) as float32

        Raises:
            VectorIndexError: If vector has wrong dimension or type
        """
        # Validate input
        if feature_vector.shape != (self.dimension,):
            raise VectorIndexError(
                f"Expected vector shape ({self.dimension},), got {feature_vector.shape}"
            )

        if feature_vector.dtype != np.float32:
            feature_vector = feature_vector.astype(np.float32)

        # FAISS requires 2D array: (n_vectors, dimension)
        vector_2d = feature_vector.reshape(1, -1)

        # Get next index position
        next_idx = self.index.ntotal

        # Add to FAISS index
        self.index.add(vector_2d)

        # Store UUID mapping
        self.id_map[next_idx] = pose_id

        logger.debug(
            f"Added vector to index: pose_id={pose_id}, idx={next_idx}",
            extra={"extra_data": {"pose_id": str(pose_id), "index_size": self.size()}},
        )

    def update(self, pose_id: UUID, feature_vector: np.ndarray) -> bool:
        """
        Update feature vector in index.

        This is implemented as remove + add since FAISS doesn't support
        direct updates.

        Args:
            pose_id: UUID of pose to update
            feature_vector: New feature vector (dimension,)

        Returns:
            True if updated, False if pose not found in index
        """
        # Check if pose exists
        found = any(pid == pose_id for pid in self.id_map.values())

        if not found:
            logger.warning(f"Pose {pose_id} not found in index for update")
            return False

        # Remove old vector
        self.remove(pose_id)

        # Add new vector
        self.add(pose_id, feature_vector)

        logger.debug(f"Updated vector in index: pose_id={pose_id}")
        return True

    def remove(self, pose_id: UUID) -> bool:
        """
        Remove feature vector from index.

        Note: FAISS IndexFlatL2 doesn't support direct removal, so we rebuild
        the index excluding the target vector. Uses batched processing to avoid
        memory leaks when rebuilding large indices.

        Args:
            pose_id: UUID of pose to remove

        Returns:
            True if removed, False if not found
        """
        import gc

        # Find internal index
        internal_idx = None
        for idx, pid in self.id_map.items():
            if pid == pose_id:
                internal_idx = idx
                break

        if internal_idx is None:
            logger.warning(f"Pose {pose_id} not found in index for removal")
            return False

        # Store old index and map references
        old_index = self.index
        old_map = self.id_map.copy()

        try:
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_map = {}

            # Rebuild with batched processing to avoid memory spike
            BATCH_SIZE = 1000
            new_idx = 0

            for batch_start in range(0, old_index.ntotal, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, old_index.ntotal)

                # Process batch
                batch_vectors = []
                batch_ids = []

                for idx in range(batch_start, batch_end):
                    if idx == internal_idx:
                        continue  # Skip the vector to remove

                    # Reconstruct vector from old index
                    vector = old_index.reconstruct(int(idx))
                    batch_vectors.append(vector)
                    batch_ids.append(old_map[idx])

                # Add batch to new index
                if batch_vectors:
                    vectors_array = np.vstack(batch_vectors)
                    self.index.add(vectors_array)

                    # Update ID map
                    for bid in batch_ids:
                        self.id_map[new_idx] = bid
                        new_idx += 1

                    # Clean up batch arrays
                    del batch_vectors, batch_ids, vectors_array

            # Explicitly delete old index to free C++ memory
            del old_index, old_map
            gc.collect()

            logger.info(
                f"Removed pose {pose_id} from index, rebuilt with {new_idx} vectors"
            )
            return True

        except Exception as e:
            # Restore on error
            self.index = old_index
            self.id_map = old_map
            logger.error(f"Failed to remove pose from index: {e}")
            raise

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[SimilarPose]:
        """
        Find k most similar poses.

        Args:
            query_vector: Query feature vector (dimension,)
            k: Number of nearest neighbors to return

        Returns:
            List of SimilarPose objects, ordered by similarity (most similar first)

        Raises:
            VectorIndexError: If query vector has wrong shape
        """
        if self.size() == 0:
            logger.warning("Search called on empty index")
            return []

        # Validate query
        if query_vector.shape != (self.dimension,):
            raise VectorIndexError(
                f"Query vector shape {query_vector.shape} doesn't match index dimension {self.dimension}"
            )

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Limit k to index size
        k = min(k, self.size())

        # FAISS search expects 2D query: (n_queries, dimension)
        query_2d = query_vector.reshape(1, -1)

        # Search returns (distances, indices) both of shape (n_queries, k)
        distances, indices = self.index.search(query_2d, k)

        # Convert to SimilarPose objects
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 for "not found" when k > index size
            if idx == -1:
                continue

            pose_id = self.id_map[int(idx)]

            # Convert L2 distance to similarity score in [0, 1]
            # Using exponential decay: similarity = exp(-distance)
            # This maps distance=0 to similarity=1, and distance→∞ to similarity→0
            similarity_score = np.exp(-float(dist))

            results.append(
                SimilarPose(
                    pose_id=pose_id,
                    distance=float(dist),
                    similarity_score=similarity_score,
                )
            )

        logger.debug(
            f"Search found {len(results)} similar poses",
            extra={
                "extra_data": {
                    "k": k,
                    "results_count": len(results),
                    "top_distance": results[0].distance if results else None,
                }
            },
        )

        return results

    def size(self) -> int:
        """Get number of vectors in index."""
        return int(self.index.ntotal)

    def save(self) -> None:
        """
        Persist index to disk.

        Saves both the FAISS index and the ID mapping.
        """
        if self.index_path is None:
            logger.warning("No index_path set, cannot save")
            return

        # Create directory if needed
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save ID mapping
        map_path = self.index_path.with_suffix(".pkl")
        with open(map_path, "wb") as f:
            pickle.dump(self.id_map, f)

        logger.info(
            f"Saved index to {self.index_path}",
            extra={"extra_data": {"size": self.size(), "path": str(self.index_path)}},
        )

    def load(self) -> None:
        """
        Load index from disk.

        Raises:
            VectorIndexError: If files don't exist or are incompatible
        """
        if self.index_path is None:
            raise VectorIndexError("No index_path set, cannot load")

        if not self.index_path.exists():
            raise VectorIndexError(f"Index file not found: {self.index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # Validate dimension
        if self.index.d != self.dimension:
            raise VectorIndexError(
                f"Loaded index has dimension {self.index.d}, expected {self.dimension}"
            )

        # Load ID mapping
        map_path = self.index_path.with_suffix(".pkl")
        if not map_path.exists():
            raise VectorIndexError(f"ID map file not found: {map_path}")

        with open(map_path, "rb") as f:
            self.id_map = pickle.load(f)

        logger.info(
            f"Loaded index from {self.index_path}",
            extra={"extra_data": {"size": self.size()}},
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for monitoring."""
        return {
            "total_vectors": self.size(),
            "dimension": self.dimension,
            "index_type": "IndexFlatL2",
            "memory_usage_mb": self.size()
            * self.dimension
            * 4
            / (1024 * 1024),  # float32 = 4 bytes
        }
