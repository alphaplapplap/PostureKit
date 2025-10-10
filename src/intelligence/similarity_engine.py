"""
FAISS-based pose similarity search engine.

Provides fast nearest-neighbor search across geometric pose features.
Supports batch indexing, incremental updates, and persistence.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
import faiss
from uuid import UUID

from src.storage.models import GeometricFeatures, FusedFeatures, PoseDetection, Image

logger = logging.getLogger(__name__)


class SimilarityEngine:
    """FAISS-powered similarity search for pose features."""

    def __init__(self, storage_manager, index_dir: str = "data/indices"):
        """
        Initialize similarity engine.

        Args:
            storage_manager: StorageManager instance for database access
            index_dir: Directory to store FAISS index files
        """
        self.storage = storage_manager
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.pose_ids: List[str] = []  # Maps FAISS index → pose UUID
        self.index_path = self.index_dir / "pose_features.index"
        self.mapping_path = self.index_dir / "pose_mapping.npy"

        # Try to load existing index
        self.load_index()

    def build_index(
        self,
        force_rebuild: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> None:
        """
        Build FAISS index from all geometric features in database.

        Args:
            force_rebuild: If True, rebuild even if index exists
            progress_callback: Optional callback(current, total, message)
        """
        if self.index is not None and not force_rebuild:
            logger.info("Index already loaded, skipping build")
            if progress_callback:
                progress_callback(1, 1, "Index already loaded")
            return

        logger.info("Building FAISS index from database...")
        if progress_callback:
            progress_callback(0, 3, "Querying database...")

        # Get all fused features (geometric + visual) with image info for better filtering
        with self.storage.session_scope() as session:
            query = session.query(
                FusedFeatures.pose_id,
                FusedFeatures.feature_vector,
                PoseDetection.overall_confidence,
                Image.file_path
            ).join(
                PoseDetection, FusedFeatures.pose_id == PoseDetection.id
            ).join(
                Image, PoseDetection.image_id == Image.id
            ).order_by(PoseDetection.created_at)

            features = query.all()

        if not features:
            logger.warning("No features found in database")
            if progress_callback:
                progress_callback(1, 1, "No features found")
            return

        if progress_callback:
            progress_callback(1, 3, f"Processing {len(features)} poses...")

        # Prepare data for FAISS
        self.pose_ids = [str(f.pose_id) for f in features]
        vectors = np.array([f.feature_vector for f in features], dtype=np.float32)

        # Validate vectors
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            logger.warning("Found NaN or Inf values in feature vectors, filtering...")
            valid_mask = ~(np.isnan(vectors).any(axis=1) | np.isinf(vectors).any(axis=1))
            vectors = vectors[valid_mask]
            self.pose_ids = [pid for i, pid in enumerate(self.pose_ids) if valid_mask[i]]
            logger.info(f"Kept {len(self.pose_ids)}/{len(features)} valid poses")

        if progress_callback:
            progress_callback(2, 3, "Building FAISS index...")

        # Build index (L2 distance = Euclidean similarity)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)

        logger.info(f"Built FAISS index with {len(self.pose_ids)} poses")

        # Save to disk
        self.save_index()

        if progress_callback:
            progress_callback(3, 3, f"Index built with {len(self.pose_ids)} poses")

    def save_index(self) -> None:
        """Persist FAISS index and ID mapping to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        try:
            faiss.write_index(self.index, str(self.index_path))
            np.save(self.mapping_path, np.array(self.pose_ids))
            logger.info(f"Saved index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load_index(self) -> bool:
        """
        Load FAISS index from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.index_path.exists() or not self.mapping_path.exists():
            logger.info("No existing index found")
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            self.pose_ids = np.load(self.mapping_path, allow_pickle=True).tolist()
            logger.info(f"Loaded index with {len(self.pose_ids)} poses")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def search_by_feature(
        self,
        feature_vector: np.ndarray,
        k: int = 20,
        min_confidence: float = 0.0,
        deduplicate_images: bool = True
    ) -> List[Dict]:
        """
        Find k most similar poses to given feature vector.

        Args:
            feature_vector: 628-dim fused feature vector (52-dim geometric + 576-dim visual)
            k: Number of results to return
            min_confidence: Minimum detection confidence threshold
            deduplicate_images: If True, return only one pose per image

        Returns:
            List of dicts with keys: pose_id, distance, rank, metadata
        """
        if self.index is None:
            logger.error("Index not built. Call build_index() first.")
            return []

        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Ensure feature vector is correct shape and type
        query = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # Request 3x more results to account for confidence filtering and deduplication
        search_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        # Build results with database metadata
        results = []
        seen_images = set()  # Track seen images for deduplication

        with self.storage.session_scope() as session:
            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                pose_id = UUID(self.pose_ids[idx])

                # Get pose metadata
                pose_query = session.query(
                    PoseDetection,
                    Image,
                    GeometricFeatures
                ).join(
                    Image, PoseDetection.image_id == Image.id
                ).join(
                    GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id
                ).filter(
                    PoseDetection.id == pose_id
                )

                result = pose_query.first()
                if not result:
                    continue

                pose, image, features = result

                # Filter by confidence
                if pose.overall_confidence < min_confidence:
                    continue

                # Deduplicate by image
                if deduplicate_images:
                    if str(image.id) in seen_images:
                        continue
                    seen_images.add(str(image.id))

                results.append({
                    'pose_id': str(pose_id),
                    'distance': float(dist),
                    'similarity_score': self._distance_to_similarity(float(dist)),
                    'rank': len(results) + 1,  # Rank after filtering
                    'image_path': image.file_path,
                    'image_id': str(image.id),
                    'detection_confidence': pose.overall_confidence,
                    'is_corrected': pose.is_corrected,
                    'person_id': pose.person_id,
                    'category': None,  # Will be filled if training_labels exist
                    'image_width': image.width,
                    'image_height': image.height,
                    'file_size': image.file_size_bytes
                })

                # Stop once we have enough results
                if len(results) >= k:
                    break

        return results

    def search_by_pose_id(
        self,
        pose_id: UUID,
        k: int = 20,
        exclude_self: bool = True,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        Find similar poses to a given pose ID.

        Args:
            pose_id: UUID of reference pose
            k: Number of results to return
            exclude_self: If True, exclude the query pose from results
            min_confidence: Minimum detection confidence threshold

        Returns:
            List of similar poses with metadata
        """
        # Get fused feature vector for this pose
        with self.storage.session_scope() as session:
            features = session.query(FusedFeatures).filter(
                FusedFeatures.pose_id == pose_id
            ).first()

            if not features:
                logger.error(f"No fused features found for pose {pose_id}")
                return []

            query_vector = features.feature_vector

        # Search (request k+1 if excluding self)
        search_k = k + 1 if exclude_self else k
        results = self.search_by_feature(query_vector, k=search_k, min_confidence=min_confidence)

        # Remove self if requested
        if exclude_self:
            results = [r for r in results if r['pose_id'] != str(pose_id)][:k]

        return results

    def add_pose(self, pose_id: UUID) -> bool:
        """
        Add a single pose to the index (incremental update).

        Args:
            pose_id: UUID of pose to add

        Returns:
            True if successfully added
        """
        if self.index is None:
            logger.warning("Index not initialized, building new index")
            self.build_index()
            return True

        # Get fused feature vector
        with self.storage.session_scope() as session:
            features = session.query(FusedFeatures).filter(
                FusedFeatures.pose_id == pose_id
            ).first()

            if not features:
                logger.error(f"No fused features found for pose {pose_id}")
                return False

            vector = np.array(features.feature_vector, dtype=np.float32).reshape(1, -1)

        # Add to index
        self.index.add(vector)
        self.pose_ids.append(str(pose_id))

        # Save updated index
        self.save_index()

        logger.info(f"Added pose {pose_id} to index (now {len(self.pose_ids)} poses)")
        return True

    def remove_pose(self, pose_id: UUID) -> bool:
        """
        Remove a pose from the index.

        Note: FAISS IndexFlatL2 doesn't support deletion, so this rebuilds the index.
        For better performance with frequent deletions, consider using IndexIVFFlat.

        Args:
            pose_id: UUID of pose to remove

        Returns:
            True if successfully removed
        """
        if str(pose_id) not in self.pose_ids:
            logger.warning(f"Pose {pose_id} not in index")
            return False

        logger.info(f"Rebuilding index to remove pose {pose_id}")
        self.build_index(force_rebuild=True)
        return True

    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            'total_poses': len(self.pose_ids) if self.pose_ids else 0,
            'dimension': self.index.d if self.index else 0,
            'index_type': type(self.index).__name__ if self.index else None,
            'index_exists': self.index is not None,
            'index_path': str(self.index_path),
            'mapping_path': str(self.mapping_path)
        }

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        Convert L2 distance to similarity score [0, 1].

        Uses inverse exponential decay: similarity = exp(-distance / scale)
        Scale chosen based on typical geometric feature distances.
        """
        scale = 2.0  # Empirically determined for 52-dim normalized features
        return float(np.exp(-distance / scale))
