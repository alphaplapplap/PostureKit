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
import threading

from src.storage.models import GeometricFeatures, FusedFeatures, PoseDetection, Image
from src.config.model_config import SearchFeatureMode

logger = logging.getLogger(__name__)


class DimensionMismatchError(Exception):
    """Raised when query dimension doesn't match index dimension."""
    pass


class SimilarityEngine:
    """FAISS-powered similarity search for pose features."""

    def __init__(
        self,
        storage_manager,
        index_dir: str = "data/indices",
        database_profile: Optional[str] = None,
        feature_mode: SearchFeatureMode = SearchFeatureMode.GEOMETRIC_ONLY
    ):
        """
        Initialize similarity engine.

        Args:
            storage_manager: StorageManager instance for database access
            index_dir: Directory to store FAISS index files (deprecated, use database_profile)
            database_profile: Database profile (irl/2d/3d) for profile-aware index paths
            feature_mode: Feature type to use for similarity search (GEOMETRIC_ONLY or FUSED_MULTIMODAL)
        """
        from src.config.settings import settings

        self.storage = storage_manager
        self.database_profile = database_profile or getattr(storage_manager, 'database_profile', settings.DB_PROFILE)
        self.feature_mode = feature_mode

        # Use profile-aware index directory if profile is specified
        if database_profile:
            self.index_dir = settings.get_index_dir(database_profile)
        else:
            self.index_dir = Path(index_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.dimension: int = feature_mode.get_feature_dim()  # 52 for geometric, 628 for fused
        self.pose_id_map: Dict[int, str] = {}  # FAISS ID → pose UUID (for efficient deletion)
        self.next_id_counter: int = 0  # Counter for assigning stable IDs
        self.index_path = self.index_dir / "pose_features.index"
        self.mapping_path = self.index_dir / "pose_mapping.npy"

        # Tracks whether the currently-loaded index is a clean, validated load
        # or a successful build. Prevents close() from re-saving stale data when
        # load_index() fails validation or when the engine is shut down without
        # any local mutations (e.g., another process rebuilt the index externally).
        self._loaded_clean: bool = False

        # Thread safety for concurrent index operations
        self._index_lock = threading.RLock()  # Reentrant lock for nested calls
        self._build_lock = threading.Lock()    # Exclusive lock for building

        # Search results cache (LRU with TTL)
        from collections import OrderedDict
        import time as time_module
        self._search_cache: OrderedDict = OrderedDict()  # (query_hash, k) -> (results, timestamp)
        self._search_cache_max_size = 1000  # Maximum cached queries
        self._search_cache_ttl = 600  # 10 minutes TTL
        self._cache_lock = threading.Lock()  # Protects OrderedDict from concurrent access

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

        logger.info(f"Building FAISS index from database (mode: {self.feature_mode.value})...")
        if progress_callback:
            progress_callback(0, 3, "Querying database...")

        # Query features based on feature mode
        with self.storage.session_scope() as session:
            if self.feature_mode == SearchFeatureMode.GEOMETRIC_ONLY:
                # Use geometric features (52-dim, keypoint-based)
                query = session.query(
                    GeometricFeatures.pose_id,
                    GeometricFeatures.feature_vector,
                    PoseDetection.overall_confidence,
                    Image.file_path
                ).join(
                    PoseDetection, GeometricFeatures.pose_id == PoseDetection.id
                ).join(
                    Image, PoseDetection.image_id == Image.id
                ).order_by(PoseDetection.created_at)

            elif self.feature_mode == SearchFeatureMode.FUSED_MULTIMODAL:
                # Use fused features (628-dim, geometric + visual)
                query = session.query(
                    FusedFeatures.pose_id,
                    FusedFeatures.fused_vector.label('feature_vector'),
                    PoseDetection.overall_confidence,
                    Image.file_path
                ).join(
                    PoseDetection, FusedFeatures.pose_id == PoseDetection.id
                ).join(
                    Image, PoseDetection.image_id == Image.id
                ).order_by(PoseDetection.created_at)

            else:
                raise ValueError(f"Unsupported feature mode: {self.feature_mode}")

            features = query.all()

        if not features:
            logger.warning("No features found in database")
            if progress_callback:
                progress_callback(1, 1, "No features found")
            return

        if progress_callback:
            progress_callback(1, 3, f"Processing {len(features)} poses...")

        # Prepare data for FAISS with stable ID mapping
        pose_uuids = [str(f.pose_id) for f in features]
        vectors = np.array([f.feature_vector for f in features], dtype=np.float32)

        # Validate vectors
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            logger.warning("Found NaN or Inf values in feature vectors, filtering...")
            valid_mask = ~(np.isnan(vectors).any(axis=1) | np.isinf(vectors).any(axis=1))
            vectors = vectors[valid_mask]
            pose_uuids = [pid for i, pid in enumerate(pose_uuids) if valid_mask[i]]
            logger.info(f"Kept {len(pose_uuids)}/{len(features)} valid poses")

        if progress_callback:
            progress_callback(2, 3, "Building FAISS index...")

        # Build index with IndexIDMap for efficient deletion
        dimension = vectors.shape[1]
        self.dimension = dimension  # Store dimension for similarity calculation
        base_index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(base_index)
        logger.info(f"Built index with dimension: {self.dimension}")

        # Assign sequential IDs and build mapping
        ids = np.arange(len(vectors), dtype=np.int64)
        self.index.add_with_ids(vectors, ids)
        self.pose_id_map = {int(i): uuid for i, uuid in enumerate(pose_uuids)}
        self.next_id_counter = len(vectors)

        logger.info(f"Built FAISS index with {len(self.pose_id_map)} poses")

        # Save to disk
        self.save_index()
        self._loaded_clean = True

        if progress_callback:
            progress_callback(3, 3, f"Index built with {len(self.pose_id_map)} poses")

    def save_index(self) -> None:
        """Persist FAISS index and ID mapping to disk atomically."""
        if self.index is None:
            logger.warning("No index to save")
            return

        try:
            import tempfile
            import os

            # Verify consistency before save
            if self.index.ntotal != len(self.pose_id_map):
                raise ValueError(
                    f"Index/map size mismatch: index={self.index.ntotal}, map={len(self.pose_id_map)}"
                )

            # Write to temporary files first
            tmp_index = self.index_path.parent / f".{self.index_path.name}.tmp"
            # For numpy, create temp without .npy since np.save adds it automatically
            tmp_mapping_base = self.mapping_path.parent / f".{self.mapping_path.stem}.tmp"

            try:
                # Save FAISS index to temp
                faiss.write_index(self.index, str(tmp_index))

                # Save ID mapping to temp with validation data
                # np.save automatically adds .npy extension
                save_data = {
                    'pose_id_map': self.pose_id_map,
                    'next_id_counter': self.next_id_counter,
                    'index_size': self.index.ntotal,  # For validation on load
                    'feature_mode': self.feature_mode.value,  # Store feature mode
                    'dimension': self.dimension,  # Store dimension for validation
                }
                np.save(str(tmp_mapping_base), save_data)
                # np.save adds .npy to the base name, so append (not replace with .with_suffix)
                tmp_mapping = tmp_mapping_base.parent / f"{tmp_mapping_base.name}.npy"

                # Verify temp files are valid
                test_index = faiss.read_index(str(tmp_index))
                if test_index.ntotal != self.index.ntotal:
                    raise ValueError("Index verification failed after write")
                del test_index

                # Atomic renames (both or neither) - convert Path to str
                os.replace(str(tmp_index), str(self.index_path))
                os.replace(str(tmp_mapping), str(self.mapping_path))

                logger.info(f"Atomically saved index with {len(self.pose_id_map)} poses")

            finally:
                # Clean up temp files if they still exist
                # np.save creates .npy version: .pose_mapping.tmp.npy
                tmp_mapping_actual = tmp_mapping_base.parent / f"{tmp_mapping_base.name}.npy"
                for tmp_file in [tmp_index, tmp_mapping_actual]:
                    if tmp_file.exists():
                        try:
                            tmp_file.unlink()
                        except:
                            pass

        except Exception as e:
            logger.error(f"Failed to save index: {e}", exc_info=True)

    def _reset_index_state(self) -> None:
        """Clear in-memory index state. Used after failed loads so close() won't re-save stale data."""
        self.index = None
        self.pose_id_map = {}
        self.next_id_counter = 0

    def load_index(self) -> bool:
        """
        Load FAISS index from disk with backwards compatibility and corruption recovery.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.index_path.exists() or not self.mapping_path.exists():
            logger.info("No existing index found")
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            loaded_data = np.load(self.mapping_path, allow_pickle=True)

            # Handle both old (list) and new (dict) formats
            # New format: 0-dimensional array containing a dict
            # Old format: 1-dimensional array of UUIDs
            if isinstance(loaded_data, np.ndarray) and loaded_data.ndim == 0:
                # New format: dict with pose_id_map and next_id_counter
                try:
                    data_dict = loaded_data.item()
                    if isinstance(data_dict, dict) and 'pose_id_map' in data_dict:
                        self.pose_id_map = {int(k): v for k, v in data_dict['pose_id_map'].items()}
                        self.next_id_counter = data_dict['next_id_counter']

                        # Validate feature mode (if stored)
                        if 'feature_mode' in data_dict:
                            loaded_mode = data_dict['feature_mode']
                            if loaded_mode != self.feature_mode.value:
                                logger.error(
                                    f"Feature mode mismatch: index has '{loaded_mode}', "
                                    f"engine expects '{self.feature_mode.value}'. "
                                    f"Index rebuild required."
                                )
                                self._reset_index_state()
                                return False

                        # Validate dimension (if stored)
                        if 'dimension' in data_dict:
                            loaded_dim = data_dict['dimension']
                            if loaded_dim != self.dimension:
                                logger.error(
                                    f"Dimension mismatch: index has {loaded_dim}D, "
                                    f"engine expects {self.dimension}D. "
                                    f"Index rebuild required."
                                )
                                self._reset_index_state()
                                return False

                        logger.info(
                            f"Loaded index in new format with {len(self.pose_id_map)} poses "
                            f"(mode: {self.feature_mode.value}, dim: {self.dimension})"
                        )
                    else:
                        raise ValueError("Invalid dict format in mapping file")
                except Exception as e:
                    logger.error(f"Failed to load new format: {e}")
                    self._reset_index_state()
                    return False
            else:
                # Old format: list of UUIDs (backwards compatibility)
                logger.info("Converting old index format to IndexIDMap...")
                old_pose_ids = loaded_data.tolist()
                self.pose_id_map = {i: str(pid) for i, pid in enumerate(old_pose_ids)}
                self.next_id_counter = len(self.pose_id_map)

                # Upgrade index to IndexIDMap if needed
                if not isinstance(self.index, faiss.IndexIDMap):
                    dimension = self.index.d
                    n_vectors = self.index.ntotal

                    # Extract vectors using proper FAISS API
                    try:
                        # For IndexFlatL2, use reconstruct_n to get all vectors
                        old_vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
                        for i in range(n_vectors):
                            self.index.reconstruct(i, old_vectors[i])
                    except Exception as reconstruct_error:
                        logger.warning(f"Failed to extract vectors: {reconstruct_error}")
                        logger.warning("Skipping index upgrade, using existing index as-is")
                        # Can't upgrade, but old index still works
                        self._loaded_clean = True
                        return True

                    base_index = faiss.IndexFlatL2(dimension)
                    new_index = faiss.IndexIDMap(base_index)
                    ids = np.arange(len(old_vectors), dtype=np.int64)
                    new_index.add_with_ids(old_vectors, ids)
                    self.index = new_index
                    logger.info(f"Upgraded to IndexIDMap with {len(self.pose_id_map)} poses")

            # Extract and store dimension from loaded index
            if isinstance(self.index, faiss.IndexIDMap):
                self.dimension = self.index.index.d
            elif hasattr(self.index, 'd'):
                self.dimension = self.index.d
            else:
                logger.warning("Could not determine dimension from loaded index, using default 52")
                self.dimension = 52

            logger.info(f"Loaded index with dimension: {self.dimension}")

            self._loaded_clean = True
            return True
        except Exception as e:
            logger.error(f"Failed to load index (likely corrupted): {e}")
            logger.info("Deleting corrupted index files for automatic rebuild...")

            # Delete corrupted files
            try:
                if self.index_path.exists():
                    self.index_path.unlink()
                    logger.info(f"Deleted corrupted index: {self.index_path}")
                if self.mapping_path.exists():
                    self.mapping_path.unlink()
                    logger.info(f"Deleted corrupted mapping: {self.mapping_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup corrupted files: {cleanup_error}")

            self._reset_index_state()
            return False

    def search_by_feature(
        self,
        feature_vector: np.ndarray,
        query_confidence: Optional[np.ndarray] = None,
        k: int = 20,
        min_confidence: float = 0.0,
        min_feature_confidence: float = 0.35,
        min_valid_overlap: int = 12,
        deduplicate_images: bool = True,
        required_regions: Optional[List[str]] = None,
        min_region_confidence: float = 0.3,
        min_similarity: float = 0.0,
        query_keypoints: Optional[np.ndarray] = None,
        query_bbox: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Find k most similar poses to given feature vector with caching.

        Supports confidence-aware search: when query_confidence is provided,
        re-ranks FAISS results by comparing only mutually valid dimensions.

        Supports OKS-based search: when query_keypoints and query_bbox are provided,
        re-ranks using Object Keypoint Similarity metric (more accurate than L2).

        Args:
            feature_vector: Feature vector matching configured mode (52-dim geometric or 628-dim fused)
            query_confidence: Optional confidence scores (0-1) for query features (same dim as feature_vector).
                            If None, uses standard FAISS distance (backward compatible).
            k: Number of results to return
            min_confidence: Minimum detection confidence threshold (overall pose confidence)
            min_feature_confidence: Minimum confidence threshold for valid dimension (masked distance)
                                   Default 0.35 (relaxed from 0.5 to handle occlusion better)
            min_valid_overlap: Minimum valid dimensions required for comparison (masked distance)
                              Default 12 for 52-dim (~23% minimum overlap), adjust for fused mode
            deduplicate_images: If True, return only one pose per image
            required_regions: Optional list of body regions that must be visible
                             (e.g., ['face', 'feet', 'hands']). Uses NudeNet detections.
            min_region_confidence: Minimum confidence for required region visibility
            query_keypoints: Optional query pose keypoints (133, 3) for OKS-based re-ranking
            query_bbox: Optional query bounding box [x, y, w, h] for OKS-based re-ranking

        Returns:
            List of dicts with keys: pose_id, distance, rank, metadata, visible_regions
            If confidence-aware, also includes 'valid_dimensions' key.
            If OKS-enabled, also includes 'oks_similarity' key.
        """
        # Check cache first (creates hash from feature vector)
        import time
        import hashlib
        query_hash = hashlib.md5(feature_vector.tobytes()).hexdigest()

        # Include confidence in cache key if provided
        conf_hash = (
            hashlib.md5(query_confidence.tobytes()).hexdigest()[:8]
            if query_confidence is not None
            else 'none'
        )

        # OKS inputs change the ranking entirely — the same feature vector with and
        # without keypoints must not collide in the cache
        kp_hash = (
            hashlib.md5(query_keypoints.tobytes()).hexdigest()[:8]
            if query_keypoints is not None
            else 'none'
        )
        bbox_hash = (
            hashlib.md5(query_bbox.tobytes()).hexdigest()[:8]
            if query_bbox is not None
            else 'none'
        )

        # Log search configuration for debugging
        logger.info(f"[SEARCH] deduplicate_images={deduplicate_images}, k={k}, min_conf={min_confidence:.2f}, "
                   f"min_feat_conf={min_feature_confidence:.2f}, min_overlap={min_valid_overlap}")

        # Include region filter in cache key
        regions_tuple = tuple(sorted(required_regions)) if required_regions else None
        cache_key = (
            query_hash,
            conf_hash,
            kp_hash,
            bbox_hash,
            k,
            min_confidence,
            min_feature_confidence,
            min_valid_overlap,
            deduplicate_images,
            regions_tuple,
            min_region_confidence,
            min_similarity
        )

        # Thread-safe cache check
        with self._cache_lock:
            if cache_key in self._search_cache:
                cached_results, timestamp = self._search_cache[cache_key]
                cache_age = time.time() - timestamp

                if cache_age < self._search_cache_ttl:
                    # Cache hit - move to end for LRU
                    self._search_cache.move_to_end(cache_key)
                    logger.debug(f"Search cache hit (age: {cache_age:.1f}s)")
                    # Return deep copy to prevent cache pollution
                    import copy
                    return copy.deepcopy(cached_results)

                # Cache expired, will recompute
                del self._search_cache[cache_key]
                logger.debug(f"Search cache expired ({cache_age:.1f}s)")

        with self._index_lock:  # Thread-safe search
            if self.index is None:
                logger.error("Index not built. Call build_index() first.")
                return []

            if self.index.ntotal == 0:
                logger.warning("Index is empty")
                return []

            # Ensure feature vector is correct shape and type
            query = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

            # Validate query dimension matches index dimension
            if query.shape[1] != self.dimension:
                raise DimensionMismatchError(
                    f"Query dimension {query.shape[1]} doesn't match "
                    f"index dimension {self.dimension} (mode: {self.feature_mode.value}). "
                    f"Expected {self.dimension}-dim {self.feature_mode.value} features. "
                    f"Rebuild index or change feature mode in configuration."
                )

            # Request more results to account for confidence filtering, deduplication, and region filtering
            # Use higher multiplier for confidence-aware search (re-ranking reduces candidates)
            if query_confidence is not None:
                multiplier = 10  # Need more candidates for re-ranking
            elif required_regions:
                multiplier = 5
            else:
                multiplier = 3

            if min_similarity > 0.0:
                # Threshold mode: the caller wants EVERY pose at/above a similarity floor, not a
                # top-k slice. IndexFlatL2 scans the whole index per query regardless of k, so we
                # request all of it and let the threshold (not k) bound the result set.
                search_k = self.index.ntotal
            else:
                search_k = min(k * multiplier, self.index.ntotal)
            distances, indices = self.index.search(query, search_k)

        # === CONFIDENCE-AWARE RE-RANKING ===
        if query_confidence is not None:
            logger.debug(f"Re-ranking {len(indices[0])} FAISS candidates using confidence masking")

            # Collect candidate pose IDs
            candidate_ids = []
            faiss_results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                pose_id = UUID(self.pose_id_map[idx])
                candidate_ids.append(pose_id)
                faiss_results.append((idx, dist))

            if not candidate_ids:
                return []

            # Batch load feature vectors and confidences for all candidates.
            # In threshold mode search_k == ntotal, so candidate_ids spans the whole index;
            # skip the (huge) IN clause and load every row in one shot instead.
            with self.storage.session_scope() as session:
                base_conf_query = session.query(
                    GeometricFeatures.pose_id,
                    GeometricFeatures.feature_vector,
                    GeometricFeatures.feature_confidence
                )
                if len(candidate_ids) >= self.index.ntotal:
                    confidence_query = base_conf_query.all()
                else:
                    confidence_query = base_conf_query.filter(
                        GeometricFeatures.pose_id.in_(candidate_ids)
                    ).all()

                # Build lookup maps
                candidate_features = {}
                candidate_confidences = {}
                for pose_id, feat_vec, feat_conf in confidence_query:
                    candidate_features[pose_id] = np.array(feat_vec, dtype=np.float32)
                    # Handle NULL confidence (backward compatibility)
                    candidate_confidences[pose_id] = (
                        np.array(feat_conf, dtype=np.float32)
                        if feat_conf is not None
                        else np.ones(52, dtype=np.float32)
                    )

            # Re-compute distances using masked distance
            reranked_candidates = []
            for idx, faiss_dist in faiss_results:
                pose_id = UUID(self.pose_id_map[idx])

                # Skip if not in database query results
                if pose_id not in candidate_features:
                    continue

                candidate_vec = candidate_features[pose_id]
                candidate_conf = candidate_confidences[pose_id]

                # Compute masked distance
                masked_dist, valid_count = self._compute_masked_distance(
                    feature_vector,  # Query feature vector
                    query_confidence,  # Query confidence
                    candidate_vec,
                    candidate_conf,
                    min_confidence=min_feature_confidence,
                    min_valid_overlap=min_valid_overlap
                )

                # Insufficient overlap: not enough mutual signal to mask. Fall
                # back to the plain full-vector distance instead of dropping the
                # candidate — a heavily occluded query previously hit this cliff
                # for EVERY candidate and returned zero results. FAISS reports
                # squared L2; sqrt puts the fallback on the same true-L2 scale
                # as the masked distances it ranks against.
                if masked_dist == float('inf'):
                    logger.debug(f"Pose {pose_id}: insufficient valid overlap ({valid_count}/{min_valid_overlap}), falling back to unmasked distance")
                    masked_dist = float(np.sqrt(max(faiss_dist, 0.0)))

                reranked_candidates.append({
                    'faiss_idx': idx,
                    'pose_id': pose_id,
                    'distance': masked_dist,
                    'valid_dimensions': valid_count,
                    'faiss_distance': faiss_dist
                })

            # Sort by masked distance (ascending)
            reranked_candidates.sort(key=lambda x: x['distance'])

            logger.debug(f"Re-ranking reduced {len(faiss_results)} → {len(reranked_candidates)} candidates")

            # Reconstruct indices and distances arrays from re-ranked results
            indices = np.array([[c['faiss_idx'] for c in reranked_candidates[:search_k]]], dtype=np.int64)
            distances = np.array([[c['distance'] for c in reranked_candidates[:search_k]]], dtype=np.float32)

            # Store valid_dimensions for later inclusion in results
            valid_dimensions_map = {c['pose_id']: c['valid_dimensions'] for c in reranked_candidates}
        else:
            valid_dimensions_map = {}

        # ===== OKS-BASED RE-RANKING (Optional) =====
        from src.config.settings import settings
        oks_similarity_map = {}
        # True once OKS has replaced the distance arrays: distances are then 1-OKS in [0, 1]
        # rather than L2, and Phase 1 must convert them to similarity accordingly.
        oks_active = False

        if settings.ENABLE_OKS_METRIC and query_keypoints is not None and query_bbox is not None:
            logger.debug(f"Re-ranking using OKS metric on {len(indices[0])} candidates")

            # Collect candidate pose IDs for batch loading. Candidates arrive sorted
            # nearest-first (L2 or masked distance), so capping keeps the best N — an
            # uncapped re-rank in threshold mode would OKS-score the entire index.
            oks_candidates = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                if len(oks_candidates) >= settings.OKS_RERANK_CANDIDATES:
                    logger.info(
                        f"OKS re-rank capped to top {settings.OKS_RERANK_CANDIDATES} candidates"
                    )
                    break
                pose_id = UUID(self.pose_id_map[idx])
                oks_candidates.append({'faiss_idx': idx, 'pose_id': pose_id, 'faiss_dist': dist})

            if oks_candidates:
                # Batch load keypoints and bboxes for all candidates
                candidate_ids = [c['pose_id'] for c in oks_candidates]

                with self.storage.session_scope() as session:
                    oks_query = session.query(
                        PoseDetection.id,
                        PoseDetection.keypoints,
                        PoseDetection.bbox
                    ).filter(
                        PoseDetection.id.in_(candidate_ids)
                    ).all()

                    # Build lookup map
                    candidate_data = {
                        pose_id: (keypoints, bbox)
                        for pose_id, keypoints, bbox in oks_query
                    }

                # Re-compute distances using OKS
                oks_reranked = []
                for candidate in oks_candidates:
                    pose_id = candidate['pose_id']

                    if pose_id not in candidate_data:
                        continue

                    cand_keypoints, cand_bbox = candidate_data[pose_id]
                    cand_keypoints_array = np.array(cand_keypoints).reshape(133, 3)
                    cand_bbox_array = np.array(cand_bbox)

                    # Compute OKS distance
                    oks_dist, oks_sim = self._compute_oks_distance(
                        query_keypoints,
                        cand_keypoints_array,
                        query_bbox,
                        cand_bbox_array,
                        min_confidence=min_feature_confidence
                    )

                    # Skip if OKS failed (returns inf)
                    if oks_dist == float('inf'):
                        continue

                    oks_reranked.append({
                        'faiss_idx': candidate['faiss_idx'],
                        'pose_id': pose_id,
                        'distance': oks_dist,
                        'oks_similarity': oks_sim,
                        'faiss_distance': candidate['faiss_dist']
                    })

                # Sort by OKS distance (ascending, lower = better)
                oks_reranked.sort(key=lambda x: x['distance'])

                logger.info(f"OKS re-ranking: {len(oks_candidates)} → {len(oks_reranked)} candidates")

                # Reconstruct indices and distances from OKS-ranked results
                indices = np.array([[c['faiss_idx'] for c in oks_reranked[:search_k]]], dtype=np.int64)
                distances = np.array([[c['distance'] for c in oks_reranked[:search_k]]], dtype=np.float32)

                # Store OKS similarities for later inclusion in results
                oks_similarity_map = {c['pose_id']: c['oks_similarity'] for c in oks_reranked}
                oks_active = True

        # Build results with database metadata.
        #
        # Two-phase to avoid one DB round-trip per candidate (the old hot loop issued a query
        # per pose, which is what made large / "All" searches take ~90s):
        #   Phase 1: trim candidates cheaply by base similarity (derived from distance, no DB).
        #   Phase 2: bulk-fetch poses/images/features and body parts for the survivors in a
        #            couple of chunked queries, then iterate in memory applying the same
        #            filters plus the min_similarity floor.
        from src.config.settings import settings

        # A candidate can only reach final_similarity >= min_similarity if its base similarity
        # (pre-plausibility) is >= min_similarity / max_boost. plausibility_score <= 1.0, so the
        # boost is bounded; use that bound to stop scanning once distances grow too large
        # (candidates are already sorted nearest-first by L2 / masked / OKS distance).
        if settings.ENABLE_PLAUSIBILITY_SCORING:
            max_boost = 0.8 + 0.2 * settings.PLAUSIBILITY_WEIGHT
        else:
            max_boost = 1.0
        base_similarity_floor = (min_similarity / max_boost) if min_similarity > 0.0 else 0.0

        # In top-k mode (no threshold) bound how many candidates we bulk-load so we don't fetch
        # thumbnails for the entire re-ranking buffer; a small multiple of k covers filter
        # dropouts. In threshold mode the base-similarity early-stop bounds the shortlist instead.
        result_fetch_cap = None if min_similarity > 0.0 else max(k * 2, 200)

        # Phase 1: ordered candidate shortlist (faiss_idx, distance, pose_id, base_similarity)
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            dist_f = float(dist)
            # OKS distances are already 1-OKS in [0, 1]; mapping them through the L2
            # exp(-d/scale) curve would compress every score into [0.61, 1.0] and break
            # threshold semantics. Either way distances ascend, so similarity descends
            # and the floor early-stop below stays valid.
            base_sim = (1.0 - dist_f) if oks_active else self._distance_to_similarity(dist_f)
            if min_similarity > 0.0 and base_sim < base_similarity_floor:
                # Sorted nearest-first → every remaining candidate is also below the floor.
                break
            candidates.append((int(idx), dist_f, UUID(self.pose_id_map[idx]), base_sim))
            if result_fetch_cap is not None and len(candidates) >= result_fetch_cap:
                break

        results = []

        # Excluded folders filter (already-indexed images vanish from results without a
        # re-index). MUST be loaded before the Phase 2 session opens: get_excluded_folders
        # uses its own session_scope, and nesting it inside the active scoped session would
        # close that session and detach every ORM object Phase 2/3 is iterating.
        excluded_folders = [f['folder_path'] for f in self.storage.get_excluded_folders()]

        if candidates:
            seen_images = set()  # Track seen images for deduplication
            candidate_pose_ids = [c[2] for c in candidates]

            from src.storage.models import BodyPart
            from collections import defaultdict

            with self.storage.session_scope() as session:
                # Phase 2a: bulk-load pose/image/feature rows (chunked to keep IN lists sane)
                record_map = {}
                for chunk_start in range(0, len(candidate_pose_ids), 1000):
                    chunk = candidate_pose_ids[chunk_start:chunk_start + 1000]
                    rows = session.query(
                        PoseDetection,
                        Image,
                        GeometricFeatures
                    ).join(
                        Image, PoseDetection.image_id == Image.id
                    ).join(
                        GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id
                    ).filter(
                        PoseDetection.id.in_(chunk)
                    ).all()
                    for pose, image, features in rows:
                        record_map[pose.id] = (pose, image, features)

                # Phase 2b: bulk-load body parts for all involved images, grouped by
                # (image_id, person_index) — matches the old per-pose filter.
                image_ids = list({image.id for (_, image, _) in record_map.values()})
                bp_map = defaultdict(list)
                for chunk_start in range(0, len(image_ids), 1000):
                    chunk = image_ids[chunk_start:chunk_start + 1000]
                    for bp in session.query(BodyPart).filter(
                        BodyPart.image_id.in_(chunk),
                        BodyPart.confidence >= min_region_confidence
                    ).all():
                        bp_map[(bp.image_id, bp.person_index)].append(bp)

                # Phase 3: iterate candidates in ranked order, filter, score, cap at k.
                for faiss_idx, dist_f, pose_id, base_sim in candidates:
                    rec = record_map.get(pose_id)
                    if rec is None:
                        continue
                    pose, image, features = rec

                    # Filter by confidence
                    if pose.overall_confidence < min_confidence:
                        continue

                    # Skip images inside excluded folders
                    if excluded_folders and self.storage.is_path_excluded(image.file_path, excluded_folders):
                        continue

                    # Skip if image file no longer exists on disk
                    if not Path(image.file_path).exists():
                        logger.debug(f"Skipping missing file: {image.file_path}")
                        continue

                    # Deduplicate by image
                    if deduplicate_images:
                        if str(image.id) in seen_images:
                            continue
                        seen_images.add(str(image.id))

                    # Body parts for this image+person (already confidence-filtered in the query)
                    body_parts = bp_map.get((image.id, pose.person_id), [])

                    # Get set of visible canonical regions
                    visible_regions = list(set(bp.canonical_region for bp in body_parts))

                    # Get detailed breakdown of body parts (18 NudeNet classes)
                    visible_regions_detailed = [
                        {
                            'part_name': bp.part_name,
                            'canonical_region': bp.canonical_region,
                            'confidence': float(bp.confidence),
                            'is_exposed': bp.is_exposed
                        }
                        for bp in body_parts
                    ]

                    # Filter by required body regions/classes (if specified)
                    if required_regions:
                        # Support both canonical regions AND specific NudeNet class names
                        detected_classes = {bp.part_name for bp in body_parts}
                        detected_canonical = set(visible_regions)
                        missing = set(required_regions) - (detected_classes | detected_canonical)
                        if missing:
                            logger.debug(f"Skipping pose {pose_id}: missing {missing}")
                            continue

                    # Normalize keypoints to thumbnail size (200x200) for overlay rendering.
                    # Thumbnails FIT WITHIN a 200x200 canvas (letterbox/pillarbox), matching
                    # thumbnail_generator.py.
                    keypoints_normalized = []
                    if pose.keypoints and len(pose.keypoints) == 399:  # 133 keypoints × 3
                        width = image.width
                        height = image.height
                        scale = min(200.0 / width, 200.0 / height)
                        new_width = width * scale
                        new_height = height * scale
                        x_offset = (200.0 - new_width) / 2
                        y_offset = (200.0 - new_height) / 2
                        for i in range(0, 399, 3):
                            x = pose.keypoints[i]
                            y = pose.keypoints[i + 1]
                            conf = pose.keypoints[i + 2]
                            keypoints_normalized.append([x * scale + x_offset, y * scale + y_offset, conf])

                    # Base similarity already computed from distance in Phase 1
                    base_similarity = base_sim

                    # Compute plausibility score (anatomical validity)
                    plausibility_score = self._compute_plausibility_score(pose, features)

                    # Apply plausibility boost if enabled
                    if settings.ENABLE_PLAUSIBILITY_SCORING:
                        boost_factor = 0.8 + 0.2 * plausibility_score * settings.PLAUSIBILITY_WEIGHT
                        final_similarity = base_similarity * boost_factor
                    else:
                        final_similarity = base_similarity

                    # Apply the similarity threshold (exact, after the plausibility boost)
                    if min_similarity > 0.0 and final_similarity < min_similarity:
                        continue

                    result_dict = {
                        'pose_id': str(pose_id),
                        'distance': dist_f,
                        'similarity_score': final_similarity,
                        'base_similarity': base_similarity,  # Original score before plausibility
                        'plausibility_score': plausibility_score,  # Anatomical validity (0-1)
                        'rank': len(results) + 1,  # Rank after filtering
                        'image_path': image.file_path,
                        'image_id': str(image.id),
                        'detection_confidence': pose.overall_confidence,
                        'is_corrected': pose.is_corrected,
                        'person_id': pose.person_id,
                        'bbox': pose.bbox,  # Bounding box [x_min, y_min, x_max, y_max]
                        'category': None,  # Will be filled if training_labels exist
                        'image_width': image.width,
                        'image_height': image.height,
                        'file_size': image.file_size_bytes,
                        'thumbnail': image.thumbnail,  # Pre-generated JPEG thumbnail bytes
                        'visible_regions': visible_regions,  # Canonical regions (7 categories)
                        'visible_regions_detailed': visible_regions_detailed,  # Full NudeNet classes (18 categories)
                        'keypoints': keypoints_normalized  # Normalized to 200x200 for skeleton overlay
                    }

                    # Add valid_dimensions if confidence-aware search was used
                    if pose_id in valid_dimensions_map:
                        result_dict['valid_dimensions'] = valid_dimensions_map[pose_id]

                    # Add OKS similarity if OKS-based search was used
                    if pose_id in oks_similarity_map:
                        result_dict['oks_similarity'] = oks_similarity_map[pose_id]

                    results.append(result_dict)

                    # Stop once we have enough results (k is large in threshold mode)
                    if len(results) >= k:
                        break

        # The plausibility boost multiplies each score by a per-pose factor in [0.8, 1.0],
        # so final similarity is no longer monotonic in distance — re-sort so implausible
        # poses sink in position, not just in score. (Top-k caveat, pre-existing: in top-k
        # mode the len(results) >= k cut above happens in distance order, so a boosted pose
        # just outside the buffer can't re-enter; threshold mode is unaffected since k≈ntotal.)
        results.sort(key=lambda r: r['similarity_score'], reverse=True)
        for rank, result_dict in enumerate(results, 1):
            result_dict['rank'] = rank

        # Thread-safe cache store and eviction
        with self._cache_lock:
            self._search_cache[cache_key] = (results, time.time())

            # Enforce cache size limit (LRU eviction)
            if len(self._search_cache) > self._search_cache_max_size:
                # Remove oldest entry (first item in OrderedDict)
                self._search_cache.popitem(last=False)
                logger.debug("Evicted oldest search cache entry")

        return results

    def search_by_pose_id(
        self,
        pose_id: UUID,
        k: int = 20,
        exclude_self: bool = True,
        min_confidence: float = 0.0,
        min_feature_confidence: float = 0.35,
        min_valid_overlap: int = 12,
        required_regions: Optional[List[str]] = None,
        min_region_confidence: float = 0.3
    ) -> List[Dict]:
        """
        Find similar poses to a given pose ID.

        Args:
            pose_id: UUID of reference pose
            k: Number of results to return
            exclude_self: If True, exclude the query pose from results
            min_confidence: Minimum detection confidence threshold (overall pose confidence)
            min_feature_confidence: Minimum confidence threshold for valid dimension (masked distance)
                                   Default 0.35 (relaxed from 0.5 to handle occlusion better)
            min_valid_overlap: Minimum valid dimensions required for comparison (masked distance)
                              Default 12 for 52-dim (~23% minimum overlap)
            required_regions: Optional list of body regions that must be visible
            min_region_confidence: Minimum confidence for required region visibility

        Returns:
            List of similar poses with metadata
        """
        # Get feature vector based on configured feature mode
        with self.storage.session_scope() as session:
            if self.feature_mode == SearchFeatureMode.GEOMETRIC_ONLY:
                # Use geometric features (52-dim)
                features = session.query(GeometricFeatures).filter(
                    GeometricFeatures.pose_id == pose_id
                ).first()

                if not features:
                    logger.error(f"No geometric features found for pose {pose_id}")
                    return []

                query_vector = np.array(features.feature_vector, dtype=np.float32)

                # Load confidence (handle NULL for backward compatibility)
                query_confidence = (
                    np.array(features.feature_confidence, dtype=np.float32)
                    if features.feature_confidence is not None
                    else None  # Don't use confidence-aware search if NULL
                )

            elif self.feature_mode == SearchFeatureMode.FUSED_MULTIMODAL:
                # Use fused features (628-dim)
                features = session.query(FusedFeatures).filter(
                    FusedFeatures.pose_id == pose_id
                ).first()

                if not features:
                    logger.error(f"No fused features found for pose {pose_id}")
                    return []

                query_vector = np.array(features.fused_vector, dtype=np.float32)
                query_confidence = None  # Fused features don't have confidence scores

            else:
                raise ValueError(f"Unsupported feature mode: {self.feature_mode}")

        # Search with confidence (request k+1 if excluding self)
        search_k = k + 1 if exclude_self else k
        results = self.search_by_feature(
            query_vector,
            query_confidence=query_confidence,
            k=search_k,
            min_confidence=min_confidence,
            min_feature_confidence=min_feature_confidence,
            min_valid_overlap=min_valid_overlap,
            required_regions=required_regions,
            min_region_confidence=min_region_confidence
        )

        # Remove self if requested
        if exclude_self:
            results = [r for r in results if r['pose_id'] != str(pose_id)][:k]

        return results

    def add_pose(self, pose_id: UUID) -> bool:
        """
        Add a single pose to the index (incremental update with stable ID).

        Args:
            pose_id: UUID of pose to add

        Returns:
            True if successfully added
        """
        if self.index is None:
            logger.warning("Index not initialized, building new index")
            self.build_index()
            return True

        # Get geometric feature vector
        with self.storage.session_scope() as session:
            features = session.query(GeometricFeatures).filter(
                GeometricFeatures.pose_id == pose_id
            ).first()

            if not features:
                logger.error(f"No geometric features found for pose {pose_id}")
                return False

            vector = np.array(features.feature_vector, dtype=np.float32).reshape(1, -1)

        # Thread-safe index update
        with self._index_lock:
            # Assign new stable ID and add to index
            new_id = self.next_id_counter
            self.index.add_with_ids(vector, np.array([new_id], dtype=np.int64))
            self.pose_id_map[new_id] = str(pose_id)
            self.next_id_counter += 1

            # Clear search cache (results are now stale)
            self.clear_search_cache()

            # Save updated index
            self.save_index()
            self._loaded_clean = True

        logger.info(f"Added pose {pose_id} to index (now {len(self.pose_id_map)} poses)")
        return True

    def remove_pose(self, pose_id: UUID) -> bool:
        """
        Remove a pose from the index using efficient IndexIDMap deletion.

        Args:
            pose_id: UUID of pose to remove

        Returns:
            True if successfully removed
        """
        pose_id_str = str(pose_id)

        # Thread-safe index modification
        with self._index_lock:
            # Find FAISS ID for this pose UUID
            faiss_id = None
            for fid, uuid in self.pose_id_map.items():
                if uuid == pose_id_str:
                    faiss_id = fid
                    break

            if faiss_id is None:
                logger.warning(f"Pose {pose_id} not in index")
                return False

            # Efficient deletion using IndexIDMap
            self.index.remove_ids(np.array([faiss_id], dtype=np.int64))
            del self.pose_id_map[faiss_id]

            # Clear search cache (results are now stale)
            self.clear_search_cache()

            # Save updated index
            self.save_index()
            self._loaded_clean = True

        logger.info(f"Removed pose {pose_id} from index (now {len(self.pose_id_map)} poses)")
        return True

    def clear_search_cache(self) -> None:
        """Clear the search results cache (thread-safe)."""
        with self._cache_lock:
            self._search_cache.clear()
        logger.debug("Search cache cleared")

    def close(self) -> None:
        """
        Clean shutdown: clear caches, release locks.

        Does NOT re-save the index on close. All mutation paths (build_index,
        add_pose, remove_pose) already persist immediately, so re-saving here
        is redundant. More importantly, a blind re-save would overwrite any
        externally-rebuilt index on disk with whatever stale state this
        process happens to be holding (e.g., a subprocess that loaded an
        old-mode index before the user rebuilt it with a different config).
        """
        logger.info("Closing SimilarityEngine...")

        try:
            # Clear search cache
            self.clear_search_cache()

            logger.info("SimilarityEngine closed successfully")

        except Exception as e:
            logger.error(f"Error during SimilarityEngine close: {e}", exc_info=True)

    def unload_index(self) -> None:
        """
        Unload index from memory to free resources.

        Keeps saved index files on disk for future loading.
        Thread-safe operation using index lock.
        """
        with self._index_lock:
            if self.index is not None:
                del self.index
                self.index = None
                self.pose_id_map.clear()
                self.next_id_counter = 0
                self.clear_search_cache()  # Clear search cache when unloading
                logger.info("FAISS index unloaded from memory")
            else:
                logger.debug("No index to unload")

    def clear_index(self, delete_files: bool = False) -> None:
        """
        Clear index from memory and optionally delete saved files.

        Args:
            delete_files: If True, also delete index files from disk

        Thread-safe operation using index lock.
        """
        with self._index_lock:
            # Clear from memory
            self.index = None
            self.pose_id_map.clear()
            self.next_id_counter = 0

            # Optionally delete files
            if delete_files:
                if self.index_path.exists():
                    self.index_path.unlink()
                    logger.info(f"Deleted index file: {self.index_path}")

                if self.mapping_path.exists():
                    self.mapping_path.unlink()
                    logger.info(f"Deleted mapping file: {self.mapping_path}")

            logger.info(f"Index cleared (files {'deleted' if delete_files else 'preserved'})")

    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            'total_poses': len(self.pose_id_map),
            'dimension': self.index.d if self.index else 0,
            'index_type': type(self.index).__name__ if self.index else None,
            'index_exists': self.index is not None,
            'index_path': str(self.index_path),
            'mapping_path': str(self.mapping_path),
            'next_id_counter': self.next_id_counter
        }

    def _compute_masked_distance(
        self,
        query_vec: np.ndarray,
        query_conf: np.ndarray,
        candidate_vec: np.ndarray,
        candidate_conf: np.ndarray,
        min_confidence: float = 0.5,
        min_valid_overlap: int = 20
    ) -> tuple:
        """
        Compute L2 distance using only mutually valid dimensions.

        Only compares dimensions where BOTH poses have confidence >= threshold.
        This prevents false matches from occluded/default features.

        Args:
            query_vec: Query feature vector (52,)
            query_conf: Query confidence scores (52,)
            candidate_vec: Candidate feature vector (52,)
            candidate_conf: Candidate confidence scores (52,)
            min_confidence: Minimum confidence threshold for valid dimension
            min_valid_overlap: Minimum valid dimensions required

        Returns:
            Tuple of (masked_distance, valid_dimension_count)
            - masked_distance: L2 distance on valid dimensions only (inf if insufficient overlap)
            - valid_dimension_count: Number of dimensions used in comparison
        """
        # Find mutually valid dimensions
        valid_mask = (query_conf >= min_confidence) & (candidate_conf >= min_confidence)
        valid_count = int(np.sum(valid_mask))

        # Require minimum overlap to prevent nonsensical comparisons
        if valid_count < min_valid_overlap:
            return (float('inf'), valid_count)

        # Compute L2 distance on valid dimensions only, scaled to the
        # full-dimension equivalent (RMS-preserving). Without the scaling,
        # fewer valid dims systematically yields smaller raw distances, so
        # heavily-occluded candidates rank above better full matches purely
        # by having less signal to compare. _distance_to_similarity is
        # calibrated for the full dimension count.
        diff = query_vec[valid_mask] - candidate_vec[valid_mask]
        distance = float(np.linalg.norm(diff)) * float(
            np.sqrt(len(query_vec) / valid_count)
        )

        return (distance, valid_count)

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert L2 distance to similarity score [0, 1].

        Uses inverse exponential decay: similarity = exp(-distance / scale)
        Scale dynamically adjusted based on feature dimensionality.

        For L2 distance, typical distances scale with sqrt(dimension).
        Base scale of 2.0 was empirically determined for 52-dim features.
        """
        # Use stored dimension (set during index build)
        dimension = self.dimension

        # Scale proportional to sqrt(dimension) for L2 distance
        # This normalizes similarity scores across different dimensionalities
        base_scale = 2.0
        scale = base_scale * np.sqrt(dimension / 52.0)

        similarity = float(np.exp(-distance / scale))

        # Log first result for debugging (avoid spam)
        if not hasattr(self, '_logged_similarity'):
            logger.info(f"Similarity calculation: distance={distance:.2f} → similarity={similarity:.2%} (dim={dimension}, scale={scale:.2f})")
            self._logged_similarity = True

        return similarity

    def _compute_plausibility_score(
        self,
        pose_detection: 'PoseDetection',
        geometric_features: Optional['GeometricFeatures'] = None
    ) -> float:
        """
        Compute anatomical plausibility score for a pose (0-1).

        Checks for biomechanically valid poses:
        - Limb length symmetry (left ≈ right)
        - Joint angle validity (within anatomical limits)
        - Bilateral proportion consistency

        Args:
            pose_detection: PoseDetection model instance
            geometric_features: Optional GeometricFeatures for detailed analysis

        Returns:
            Plausibility score (0-1), where 1.0 = highly plausible
        """
        from src.config.settings import settings

        if not settings.ENABLE_PLAUSIBILITY_SCORING:
            return 1.0  # Disabled, return neutral score

        # Initialize scores
        symmetry_score = 1.0
        angle_validity_score = 1.0
        proportion_score = 1.0

        # Extract keypoints (133 × 3: x, y, confidence)
        keypoints = np.array(pose_detection.keypoints).reshape(133, 3)

        # Helper to get keypoint with confidence check
        def get_kp(idx: int, min_conf: float = 0.3) -> Optional[np.ndarray]:
            if keypoints[idx, 2] >= min_conf:
                return keypoints[idx, :2]
            return None

        # ===== 1. Limb Length Symmetry (40% weight) =====
        # Check if left/right limbs have similar lengths
        symmetry_checks = []

        # Upper arms (shoulder → elbow)
        left_shoulder, right_shoulder = get_kp(5), get_kp(6)
        left_elbow, right_elbow = get_kp(7), get_kp(8)
        if all(v is not None for v in [left_shoulder, right_shoulder, left_elbow, right_elbow]):
            left_upper_arm = float(np.linalg.norm(left_elbow - left_shoulder))
            right_upper_arm = float(np.linalg.norm(right_elbow - right_shoulder))
            if max(left_upper_arm, right_upper_arm) > 0:
                ratio = min(left_upper_arm, right_upper_arm) / max(left_upper_arm, right_upper_arm)
                symmetry_checks.append(ratio)

        # Forearms (elbow → wrist)
        left_wrist, right_wrist = get_kp(9), get_kp(10)
        if all(v is not None for v in [left_elbow, right_elbow, left_wrist, right_wrist]):
            left_forearm = float(np.linalg.norm(left_wrist - left_elbow))
            right_forearm = float(np.linalg.norm(right_wrist - right_elbow))
            if max(left_forearm, right_forearm) > 0:
                ratio = min(left_forearm, right_forearm) / max(left_forearm, right_forearm)
                symmetry_checks.append(ratio)

        # Thighs (hip → knee)
        left_hip, right_hip = get_kp(11), get_kp(12)
        left_knee, right_knee = get_kp(13), get_kp(14)
        left_thigh: Optional[float] = None
        right_thigh: Optional[float] = None
        if all(v is not None for v in [left_hip, right_hip, left_knee, right_knee]):
            left_thigh = float(np.linalg.norm(left_knee - left_hip))
            right_thigh = float(np.linalg.norm(right_knee - right_hip))
            if max(left_thigh, right_thigh) > 0:
                ratio = min(left_thigh, right_thigh) / max(left_thigh, right_thigh)
                symmetry_checks.append(ratio)

        # Shins (knee → ankle)
        left_ankle, right_ankle = get_kp(15), get_kp(16)
        if all(v is not None for v in [left_knee, right_knee, left_ankle, right_ankle]):
            left_shin = float(np.linalg.norm(left_ankle - left_knee))
            right_shin = float(np.linalg.norm(right_ankle - right_knee))
            if max(left_shin, right_shin) > 0:
                ratio = min(left_shin, right_shin) / max(left_shin, right_shin)
                symmetry_checks.append(ratio)

        if symmetry_checks:
            symmetry_score = float(np.mean(symmetry_checks))
            # Penalize if below threshold
            if symmetry_score < settings.PLAUSIBILITY_MIN_LIMB_SYMMETRY:
                symmetry_score *= 0.5  # Heavy penalty for asymmetry

        # ===== 2. Joint Angle Validity (30% weight) =====
        # Check if joint angles are within anatomical limits
        if geometric_features and hasattr(geometric_features, 'joint_angles'):
            angles = geometric_features.joint_angles
            valid_angles = []

            # Elbow: 0° (straight) to 180° (fully flexed)
            for elbow in ['left_elbow', 'right_elbow']:
                if elbow in angles and angles[elbow] is not None:
                    angle = angles[elbow]
                    # Anatomically valid range: 0-180°
                    if 0 <= angle <= 180:
                        valid_angles.append(1.0)
                    else:
                        valid_angles.append(0.5)  # Penalize invalid angles

            # Knee: 0° (straight) to 170° (flexed, can't hyperextend much)
            for knee in ['left_knee', 'right_knee']:
                if knee in angles and angles[knee] is not None:
                    angle = angles[knee]
                    # Anatomically valid range: 0-170°
                    if 0 <= angle <= 170:
                        valid_angles.append(1.0)
                    else:
                        valid_angles.append(0.5)

            if valid_angles:
                angle_validity_score = float(np.mean(valid_angles))

        # ===== 3. Body Proportion Consistency (30% weight) =====
        # Check if body proportions make sense (torso height relative to limbs).
        # NOTE: these are numpy arrays / Optional floats — explicit None checks only
        # (array truthiness raises ValueError).
        if all(v is not None for v in [left_hip, right_hip, left_shoulder, right_shoulder]):
            # Torso length (shoulder midpoint to hip midpoint)
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid = (left_hip + right_hip) / 2
            torso_length = float(np.linalg.norm(hip_mid - shoulder_mid))

            # Check torso to limb ratios
            proportion_checks = []

            # Torso should be roughly 2-3× head size, 0.8-1.2× thigh length
            # (These are approximate human body proportions)
            if left_thigh is not None and right_thigh is not None:
                avg_thigh = (left_thigh + right_thigh) / 2
                if avg_thigh > 0:
                    torso_to_thigh = torso_length / avg_thigh
                    # Expect ratio between 0.7-1.3 (torso slightly shorter than thigh)
                    if 0.7 <= torso_to_thigh <= 1.3:
                        proportion_checks.append(1.0)
                    else:
                        proportion_checks.append(0.7)  # Minor penalty for unusual proportions

            if proportion_checks:
                proportion_score = float(np.mean(proportion_checks))

        # ===== Combine Scores =====
        # Weighted average: symmetry (40%) + angles (30%) + proportions (30%)
        plausibility = (
            0.40 * symmetry_score +
            0.30 * angle_validity_score +
            0.30 * proportion_score
        )

        return float(np.clip(plausibility, 0.0, 1.0))

    def _compute_oks_distance(
        self,
        query_keypoints: np.ndarray,
        candidate_keypoints: np.ndarray,
        query_bbox: np.ndarray,
        candidate_bbox: np.ndarray,
        min_confidence: float = 0.3
    ) -> Tuple[float, float]:
        """
        Compute Object Keypoint Similarity (OKS) distance between two poses.

        OKS is the COCO evaluation metric for pose similarity, accounting for:
        - Keypoint-specific uncertainty (via sigma values)
        - Scale normalization (via bounding box area)
        - Visibility-aware matching

        Formula: OKS = Σ exp(-d²/(2s²κ²)) * δ(v>0) / Σ δ(v>0)
        Where:
        - d = Euclidean distance between keypoints
        - s = sqrt(bbox_area) (scale factor)
        - κ = keypoint-specific constant (sigma)
        - v = visibility flag

        Args:
            query_keypoints: Query pose keypoints (133, 3) [x, y, confidence]
            candidate_keypoints: Candidate pose keypoints (133, 3)
            query_bbox: Query bounding box [x, y, w, h]
            candidate_bbox: Candidate bounding box [x, y, w, h]
            min_confidence: Minimum confidence for valid keypoint

        Returns:
            Tuple of (oks_distance, oks_similarity)
            - oks_distance: 1 - OKS (for compatibility with L2 distance, lower = better)
            - oks_similarity: Raw OKS score [0, 1] (higher = better match)
        """
        from src.config.settings import settings

        if not settings.ENABLE_OKS_METRIC:
            # Fall back to standard L2 distance
            return (float('inf'), 0.0)

        # COCO keypoint sigmas (κ values) for major joints
        # Based on COCO annotation uncertainty statistics
        # Indexed by RTMW keypoint positions (first 17 match COCO)
        COCO_SIGMAS = np.array([
            0.026,  # 0: nose
            0.025,  # 1: left_eye
            0.025,  # 2: right_eye
            0.035,  # 3: left_ear
            0.035,  # 4: right_ear
            0.079,  # 5: left_shoulder
            0.079,  # 6: right_shoulder
            0.072,  # 7: left_elbow
            0.072,  # 8: right_elbow
            0.062,  # 9: left_wrist
            0.062,  # 10: right_wrist
            0.107,  # 11: left_hip
            0.107,  # 12: right_hip
            0.087,  # 13: left_knee
            0.087,  # 14: right_knee
            0.089,  # 15: left_ankle
            0.089,  # 16: right_ankle
        ], dtype=np.float32)

        # Extend sigmas for remaining RTMW keypoints (17-132)
        # Use moderate values for hand/face/foot keypoints
        extended_sigmas = np.concatenate([
            COCO_SIGMAS,  # First 17 (COCO body keypoints)
            np.full(116, 0.05, dtype=np.float32)  # Remaining 116 keypoints (hands, face, feet)
        ])

        # Canonicalize both poses into their own bbox frames (translate by bbox origin,
        # scale by own sqrt(area)). COCO's OKS compares detections within ONE image, where
        # translation is meaningful; across different images, identical poses at different
        # positions/scales would score ~0 in absolute coordinates. After canonicalization
        # distances are in normalized person-size units, so the COCO formula applies with
        # s = 1 and the κ sigmas keep their calibrated meaning.
        query_area = query_bbox[2] * query_bbox[3]  # w * h
        candidate_area = candidate_bbox[2] * candidate_bbox[3]
        if query_area <= 0 or candidate_area <= 0:
            return (float('inf'), 0.0)
        query_scale = float(np.sqrt(query_area))
        candidate_scale = float(np.sqrt(candidate_area))

        # Compute OKS
        oks_sum = 0.0
        valid_count = 0

        for i in range(min(len(query_keypoints), len(candidate_keypoints), len(extended_sigmas))):
            # Check if both keypoints are valid
            query_conf = query_keypoints[i, 2]
            cand_conf = candidate_keypoints[i, 2]

            if query_conf >= min_confidence and cand_conf >= min_confidence:
                # Euclidean distance in canonical (bbox-relative, scale-normalized) space
                qx = (query_keypoints[i, 0] - query_bbox[0]) / query_scale
                qy = (query_keypoints[i, 1] - query_bbox[1]) / query_scale
                cx = (candidate_keypoints[i, 0] - candidate_bbox[0]) / candidate_scale
                cy = (candidate_keypoints[i, 1] - candidate_bbox[1]) / candidate_scale
                dx = qx - cx
                dy = qy - cy
                d_squared = dx * dx + dy * dy

                # OKS contribution: exp(-d²/(2κ²)) — s = 1 in canonical space
                sigma = extended_sigmas[i]
                denominator = 2.0 * sigma * sigma
                oks_contribution = np.exp(-d_squared / denominator)

                oks_sum += oks_contribution
                valid_count += 1

        if valid_count == 0:
            return (float('inf'), 0.0)

        # Average OKS across valid keypoints
        oks_similarity = float(oks_sum / valid_count)

        # Convert to distance: distance = 1 - OKS
        # (so lower distance = better match, consistent with L2)
        oks_distance = float(1.0 - oks_similarity)

        return (oks_distance, oks_similarity)

    def _flip_keypoints_horizontal(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Flip keypoints horizontally (mirror across vertical axis).

        Swaps left/right keypoint pairs to find symmetric poses.
        Useful for finding mirrored poses (e.g., left arm raised ↔ right arm raised).

        Args:
            keypoints: (133, 3) array of [x, y, confidence]

        Returns:
            Flipped keypoints (133, 3) with left/right swapped
        """
        flipped = keypoints.copy()

        # RTMW-L (COCO-WholeBody) left/right swap indices
        # Format: (left_idx, right_idx) pairs
        SWAP_PAIRS = [
            # Body keypoints (COCO 17)
            (1, 2),    # left_eye ↔ right_eye
            (3, 4),    # left_ear ↔ right_ear
            (5, 6),    # left_shoulder ↔ right_shoulder
            (7, 8),    # left_elbow ↔ right_elbow
            (9, 10),   # left_wrist ↔ right_wrist
            (11, 12),  # left_hip ↔ right_hip
            (13, 14),  # left_knee ↔ right_knee
            (15, 16),  # left_ankle ↔ right_ankle
            # Face keypoints (68 points, indices 17-84)
            # Left/right face points are symmetric around centerline
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # Jawline
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # Eyes
            (31, 35), (32, 34),  # Nose
            (48, 54), (49, 53), (50, 52), (60, 64), (61, 63), (67, 65),  # Mouth
            # Left hand (indices 91-111) ↔ Right hand (indices 112-132)
            *[(91 + i, 112 + i) for i in range(21)],  # All finger points
        ]

        # Swap keypoints
        for left_idx, right_idx in SWAP_PAIRS:
            if left_idx < len(flipped) and right_idx < len(flipped):
                flipped[left_idx], flipped[right_idx] = flipped[right_idx].copy(), flipped[left_idx].copy()

        return flipped

    def search_with_flip(
        self,
        feature_vector: np.ndarray,
        keypoints: np.ndarray,
        query_confidence: Optional[np.ndarray] = None,
        k: int = 20,
        **search_kwargs
    ) -> List[Dict]:
        """
        Search for similar poses including horizontally flipped (mirrored) matches.

        Performs two searches:
        1. Normal orientation
        2. Flipped orientation (left/right swapped)

        Then merges results and marks which are mirrored matches.

        Args:
            feature_vector: Query feature vector (52-dim geometric)
            keypoints: Query keypoints (133, 3) for flipping
            query_confidence: Optional confidence scores
            k: Total number of results to return
            **search_kwargs: Additional arguments passed to search_by_feature

        Returns:
            List of dicts with 'is_flipped_match' key for mirrored results
        """
        from src.config.settings import settings

        if not settings.ENABLE_FLIP_SEARCH:
            # Flip search disabled, do normal search
            return self.search_by_feature(
                feature_vector,
                query_confidence=query_confidence,
                k=k,
                **search_kwargs
            )

        # OKS re-ranking is incompatible with flip search: the flipped leg would score the
        # UNFLIPPED query keypoints against mirror-matched candidates (penalizing exactly the
        # poses it's meant to find), and OKS distances [0, 1] are incomparable with the other
        # leg's L2 distances at merge time. Strip the OKS inputs so both legs rank by L2.
        # Mirrored-keypoint OKS for flip queries is future work.
        if search_kwargs.get('query_keypoints') is not None or search_kwargs.get('query_bbox') is not None:
            logger.info("Flip search active: OKS re-ranking disabled for this query")
        search_kwargs = {kw: v for kw, v in search_kwargs.items()
                         if kw not in ('query_keypoints', 'query_bbox')}

        # Search with normal orientation.
        # In threshold mode (min_similarity > 0) the caller wants every match above the floor,
        # so each sub-search must use the full k rather than the fixed merge-top-k.
        k_per_search = k if search_kwargs.get('min_similarity', 0.0) > 0.0 else settings.FLIP_SEARCH_MERGE_TOP_K
        normal_results = self.search_by_feature(
            feature_vector,
            query_confidence=query_confidence,
            k=k_per_search,
            **search_kwargs
        )

        # Flip keypoints and re-extract features
        flipped_keypoints = self._flip_keypoints_horizontal(keypoints)

        # Extract geometric features from flipped pose. The extractor takes a PoseResult,
        # not a raw array — derive visibility from keypoint confidence and a tight bbox
        # from the confident keypoints.
        from src.core.geometric_feature_extractor import GeometricFeatureExtractor
        from src.core.pose_detector import PoseResult

        flipped_vis = np.where(flipped_keypoints[:, 2] >= 0.3, 2, 0).astype(np.int64)
        confident = flipped_keypoints[flipped_keypoints[:, 2] >= 0.3]
        if confident.shape[0] >= 2:
            x_min, y_min = confident[:, 0].min(), confident[:, 1].min()
            flipped_bbox = np.array([
                x_min, y_min,
                confident[:, 0].max() - x_min,
                confident[:, 1].max() - y_min
            ], dtype=np.float64)
        else:
            flipped_bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        flipped_pose = PoseResult(
            keypoints=flipped_keypoints,
            visibility=flipped_vis,
            bbox=flipped_bbox,
            overall_confidence=float(flipped_keypoints[:, 2].mean()),
            person_id=0
        )

        extractor = GeometricFeatureExtractor(
            confidence_threshold=0.3,
            normalize=True
        )
        flipped_features = extractor.extract(flipped_pose)

        # Search with flipped orientation. The confidence vector must be the
        # flipped pose's own — the extractor just computed it from the mirrored
        # keypoints, so its left/right dims are already swapped to match the
        # flipped feature vector. Reusing the unflipped query_confidence masked
        # exactly the wrong side for asymmetrically occluded queries.
        flipped_results = self.search_by_feature(
            flipped_features.feature_vector,
            query_confidence=(
                flipped_features.feature_confidence
                if query_confidence is not None else None
            ),
            k=k_per_search,
            **search_kwargs
        )

        # Merge results and mark flipped matches
        # Use pose_id to deduplicate (a pose can appear in both searches)
        merged = {}

        for result in normal_results:
            pose_id = result['pose_id']
            result['is_flipped_match'] = False
            merged[pose_id] = result

        for result in flipped_results:
            pose_id = result['pose_id']
            if pose_id not in merged:
                result['is_flipped_match'] = True
                merged[pose_id] = result
            else:
                # Pose appears in both → keep better match. Compare final similarity, not
                # distance — the plausibility boost makes score non-monotonic in distance.
                if result['similarity_score'] > merged[pose_id]['similarity_score']:
                    result['is_flipped_match'] = True
                    merged[pose_id] = result

        # Sort by final similarity score (descending) and return top k
        final_results = sorted(
            merged.values(),
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:k]

        # Re-assign ranks
        for rank, result in enumerate(final_results, 1):
            result['rank'] = rank

        logger.info(f"Flip search: {len(normal_results)} normal + {len(flipped_results)} flipped → {len(final_results)} unique")

        return final_results
