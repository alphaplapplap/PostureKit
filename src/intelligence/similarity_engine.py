"""
FAISS-based pose similarity search engine.

Provides fast nearest-neighbor search across geometric pose features.
Supports batch indexing, incremental updates, and persistence.
"""

import logging
import time as _time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np
import faiss
from uuid import UUID
import threading

from src.storage.models import GeometricFeatures, FusedFeatures, PoseDetection, Image
from src.config.model_config import SearchFeatureMode
from src import constants

logger = logging.getLogger(__name__)


class DimensionMismatchError(Exception):
    """Raised when query dimension doesn't match index dimension."""
    pass


# COCO keypoint sigmas (κ values) for the 17 body joints, based on COCO annotation
# uncertainty statistics; indexed by RTMW keypoint positions (first 17 match COCO).
# Hoisted to module level (finding 3) so the vectorized OKS re-rank no longer rebuilds
# this 133-element array on every per-candidate call.
_COCO_SIGMAS = np.array([
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

# Extend sigmas for remaining RTMW keypoints (17-132): moderate value for
# hand/face/foot keypoints. Full (133,) sigma vector reused across all OKS calls.
_EXTENDED_SIGMAS = np.concatenate([
    _COCO_SIGMAS,
    np.full(116, 0.05, dtype=np.float32),
])


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
        self.uuid_to_faiss_id: Dict[str, int] = {}  # reverse map for O(1) remove_pose
        self.next_id_counter: int = 0  # Counter for assigning stable IDs
        self.index_path = self.index_dir / "pose_features.index"
        self.mapping_path = self.index_dir / "pose_mapping.npy"

        # exp(-d/scale) base, read from index metadata (save_index persists it). Lets Wave 4
        # fit the curve after re-extraction without code changes; default when absent.
        self.similarity_scale: float = constants.DEFAULT_SIMILARITY_SCALE

        # --- Confidence-aware corpus cache (findings 0 & 6) ---
        # Static between index mutations: the geometric corpus held as contiguous float32
        # matrices plus a faiss-id→row map, so the default threshold search re-ranks in one
        # broadcast numpy pass instead of a per-query full-table DB load + 75k-iteration loop.
        # Built lazily on first confidence-aware search (NOT at load — that would add ~2s
        # before the search server's {"status":"ready"} handshake), invalidated on every index
        # mutation / stale reload.
        self._corpus_feat: Optional[np.ndarray] = None    # (N, 52) float32, row order = self._corpus_faiss_ids
        self._corpus_conf: Optional[np.ndarray] = None    # (N, 52) float32 (NULL confidence → ones)
        self._corpus_faiss_ids: Optional[np.ndarray] = None  # (N,) int64 faiss ids aligned with the rows above
        self._corpus_row_of_faiss_id: Dict[int, int] = {}    # faiss id → row index
        self._corpus_lock = threading.Lock()

        # --- Batched checkpoint persistence (finding 1) ---
        # Per-pose add/remove no longer writes the full index to disk; callers persist via
        # checkpoint_index() every N mutations / T seconds and flush_index() once at the end.
        self._dirty_since_save: int = 0       # pending in-memory mutations not yet on disk
        self._last_checkpoint_time: float = _time.monotonic()
        self.checkpoint_every_n: int = 500    # poses
        self.checkpoint_every_seconds: float = 30.0

        # Tracks whether the currently-loaded index is a clean, validated load
        # or a successful build. Prevents close() from re-saving stale data when
        # load_index() fails validation or when the engine is shut down without
        # any local mutations (e.g., another process rebuilt the index externally).
        self._loaded_clean: bool = False

        # Thread safety for concurrent index operations
        self._index_lock = threading.RLock()  # Reentrant lock for nested calls
        self._build_lock = threading.Lock()    # Exclusive lock for building

        # Search results cache (LRU with TTL + byte bound)
        from collections import OrderedDict
        self._search_cache: OrderedDict = OrderedDict()  # cache_key -> (results, timestamp, approx_bytes)
        self._search_cache_max_size = 1000  # Maximum cached queries (count bound)
        self._search_cache_ttl = 600  # 10 minutes TTL
        # Byte bound: threshold-mode entries can hold the thumbnails of an entire result set
        # (~17 KB each), so a count-only cap let resident memory grow unbounded. Evict by total
        # approx bytes too. 2 GB is trivial headroom on this machine yet caps the worst case.
        self._search_cache_max_bytes = 2 * 1024 * 1024 * 1024
        self._search_cache_bytes = 0  # running sum of approx entry sizes
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
        self._rebuild_uuid_reverse_map()
        self.next_id_counter = len(vectors)
        # Corpus cache (confidence matrices) is now stale; drop it so it rebuilds lazily.
        self._invalidate_corpus_cache()

        logger.info(f"Built FAISS index with {len(self.pose_id_map)} poses")

        # Save to disk
        self.save_index()
        self._loaded_clean = True

        if progress_callback:
            progress_callback(3, 3, f"Index built with {len(self.pose_id_map)} poses")

    def save_index(self, verify: bool = True) -> None:
        """Persist FAISS index and ID mapping to disk atomically.

        Args:
            verify: When True (final saves / explicit mutations), re-read the freshly written
                index and compare ntotal as a corruption guard. Intermediate ingest checkpoints
                pass verify=False to skip a full faiss.read_index deserialization of the whole
                index per checkpoint (the DB remains source of truth and the run rebuilds the
                index once at the end, so a checkpoint need not pay the re-read cost).
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        try:
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
                    'similarity_scale': self.similarity_scale,  # exp(-d/scale) base (Wave 4 calibration)
                }
                np.save(str(tmp_mapping_base), save_data)
                # np.save adds .npy to the base name, so append (not replace with .with_suffix)
                tmp_mapping = tmp_mapping_base.parent / f"{tmp_mapping_base.name}.npy"

                # Verify temp files are valid (skipped for intermediate checkpoints)
                if verify:
                    test_index = faiss.read_index(str(tmp_index))
                    if test_index.ntotal != self.index.ntotal:
                        raise ValueError("Index verification failed after write")
                    del test_index

                # Atomic renames (both or neither) - convert Path to str
                os.replace(str(tmp_index), str(self.index_path))
                os.replace(str(tmp_mapping), str(self.mapping_path))

                # Our in-memory index IS this save — don't self-trigger the
                # staleness reload on the next search.
                self._loaded_index_mtime = self._index_files_mtime()

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

    def checkpoint_index(self, force: bool = False) -> None:
        """Persist the in-memory index to disk on a bounded interval during ingest.

        add_pose/remove_pose mutate only RAM (cheap add_with_ids / remove_ids); this writes the
        whole index at most every checkpoint_every_n mutations or checkpoint_every_seconds,
        skipping the read_index verification (verify=False) since the final flush_index() (or the
        end-of-run full rebuild) re-verifies. A crash loses at most one checkpoint window of
        additions, all recoverable from the DB. Call under no lock; takes _index_lock itself.
        """
        with self._index_lock:
            if self._dirty_since_save == 0:
                return
            now = _time.monotonic()
            if not force and self._dirty_since_save < self.checkpoint_every_n and \
                    (now - self._last_checkpoint_time) < self.checkpoint_every_seconds:
                return
            self.save_index(verify=False)
            self._loaded_clean = True
            self._dirty_since_save = 0
            self._last_checkpoint_time = now

    def flush_index(self) -> None:
        """Final, verified persistence of any pending in-memory mutations (end-of-run/cancel)."""
        with self._index_lock:
            if self._dirty_since_save == 0:
                return
            self.save_index(verify=True)
            self._loaded_clean = True
            self._dirty_since_save = 0
            self._last_checkpoint_time = _time.monotonic()

    def _rebuild_uuid_reverse_map(self) -> None:
        """Rebuild uuid→faiss_id from pose_id_map (after bulk build/load)."""
        self.uuid_to_faiss_id = {uuid: fid for fid, uuid in self.pose_id_map.items()}

    def _invalidate_corpus_cache(self) -> None:
        """Drop the cached confidence/feature corpus matrices (rebuilt lazily on next use)."""
        with self._corpus_lock:
            self._corpus_feat = None
            self._corpus_conf = None
            self._corpus_faiss_ids = None
            self._corpus_row_of_faiss_id = {}

    def _ensure_corpus_cache(self) -> bool:
        """Build the contiguous geometric corpus matrices once, keyed to the current index.

        Loads feature_vector + feature_confidence for every pose currently in pose_id_map in one
        DB query and stores them as (N, 52) float32 matrices aligned with a faiss-id array. The
        confidence-aware re-rank then masks/scores in one broadcast pass against these instead of
        re-fetching the whole table and looping per candidate each query.

        Returns True if a usable cache is present.
        """
        with self._corpus_lock:
            if self._corpus_feat is not None and self._corpus_faiss_ids is not None:
                return self._corpus_feat.shape[0] > 0
            # Snapshot the id→uuid mapping under the index lock so we cache exactly what is indexed.
            with self._index_lock:
                items = list(self.pose_id_map.items())  # (faiss_id, uuid_str)
            if not items:
                self._corpus_feat = np.zeros((0, self.dimension), dtype=np.float32)
                self._corpus_conf = np.zeros((0, self.dimension), dtype=np.float32)
                self._corpus_faiss_ids = np.zeros((0,), dtype=np.int64)
                self._corpus_row_of_faiss_id = {}
                return False

            uuid_to_fid = {uuid: int(fid) for fid, uuid in items}
            n = len(items)
            dim = self.dimension
            feat = np.zeros((n, dim), dtype=np.float32)
            conf = np.ones((n, dim), dtype=np.float32)  # NULL confidence → ones (legacy poses)
            faiss_ids = np.empty((n,), dtype=np.int64)
            row_of_fid: Dict[int, int] = {}

            row = 0
            uuids = list(uuid_to_fid.keys())
            with self.storage.session_scope() as session:
                for chunk_start in range(0, len(uuids), 1000):
                    chunk = [UUID(u) for u in uuids[chunk_start:chunk_start + 1000]]
                    rows = session.query(
                        GeometricFeatures.pose_id,
                        GeometricFeatures.feature_vector,
                        GeometricFeatures.feature_confidence
                    ).filter(GeometricFeatures.pose_id.in_(chunk)).all()
                    for pose_id, feat_vec, feat_conf in rows:
                        fid = uuid_to_fid.get(str(pose_id))
                        if fid is None:
                            continue
                        feat[row] = np.asarray(feat_vec, dtype=np.float32)
                        if feat_conf is not None:
                            conf[row] = np.asarray(feat_conf, dtype=np.float32)
                        faiss_ids[row] = fid
                        row_of_fid[fid] = row
                        row += 1

            if row < n:
                # Some indexed poses had no GeometricFeatures row (shouldn't happen for the
                # geometric mode index, but trim defensively so rows are all populated).
                feat = feat[:row]
                conf = conf[:row]
                faiss_ids = faiss_ids[:row]

            self._corpus_feat = feat
            self._corpus_conf = conf
            self._corpus_faiss_ids = faiss_ids
            self._corpus_row_of_faiss_id = row_of_fid
            logger.info(f"Built confidence-aware corpus cache: {row} poses x {dim} dims "
                        f"(~{(feat.nbytes + conf.nbytes) // (1024 * 1024)} MB)")
            return row > 0

    # Tier-1 (low-overlap) candidates are demoted by this fixed additive distance so that
    # EVERY sufficient-overlap candidate outranks EVERY fallback candidate (finding 39). The
    # offset dwarfs any realistic RMS-rescaled masked L2 (geometric distances are O(10) at
    # most), so it both guarantees the tier ordering and crushes fallback similarity to ~0 via
    # exp(-d/scale) — low-overlap placeholder geometry is no longer admitted at a positive floor.
    _FALLBACK_TIER_OFFSET = 1000.0

    def _rerank_confidence_aware(
        self,
        faiss_indices: np.ndarray,
        faiss_distances: np.ndarray,
        query_vec: np.ndarray,
        query_conf: np.ndarray,
        min_confidence: float,
        min_valid_overlap: int,
    ) -> tuple:
        """Vectorized confidence-aware re-rank over the cached corpus (findings 0, 6, 39).

        Replaces the old per-candidate Python loop + full-table re-fetch with one broadcast
        numpy pass against the contiguous corpus matrices built by _ensure_corpus_cache().

        For each candidate, the masked distance uses only dimensions where BOTH the query and
        the candidate have confidence >= min_confidence, RMS-rescaled to the full-dimension
        equivalent — numerically identical to _compute_masked_distance(). Candidates with fewer
        than min_valid_overlap mutually-valid dims are scored over whatever dims ARE valid (NOT
        the full vector with placeholder dims) and demoted to a trailing second tier.

        Args:
            faiss_indices: faiss ids for the candidate shortlist (may contain -1 padding).
            faiss_distances: squared-L2 from IndexFlatL2 (used only as a last resort when a
                candidate has zero mutually-valid dims).
            query_vec: query feature vector (dim,).
            query_conf: query per-dim confidence (dim,).
            min_confidence: per-dimension confidence floor for a "valid" dimension.
            min_valid_overlap: minimum mutually-valid dims for a tier-0 (sufficient) candidate.

        Returns:
            (faiss_idx_array, distances_array, valid_dimensions_map, fallback_pose_ids)
            - faiss_idx_array (int64, sorted nearest-first across both tiers)
            - distances_array (float32, true RMS-rescaled L2; tier-1 carries the demotion offset)
            - valid_dimensions_map: {UUID(pose_id): valid_dim_count} for every returned candidate
            - fallback_pose_ids: set of UUID(pose_id) for the demoted tier-1 candidates
        """
        empty = (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), {}, set())
        if not self._ensure_corpus_cache():
            return empty

        # Snapshot corpus references so a concurrent cache invalidation can't pull them mid-pass.
        feat = self._corpus_feat
        conf = self._corpus_conf
        row_of_fid = self._corpus_row_of_faiss_id
        if feat is None or feat.shape[0] == 0:
            return empty

        cand_fids = np.asarray(faiss_indices, dtype=np.int64).reshape(-1)
        cand_fids = cand_fids[cand_fids != -1]
        if cand_fids.size == 0:
            return empty

        # Map candidate faiss ids -> corpus rows; drop any not present (poses without a stored
        # GeometricFeatures row, which the corpus cache already trims).
        rows = np.fromiter((row_of_fid.get(int(f), -1) for f in cand_fids),
                           dtype=np.int64, count=cand_fids.size)
        keep = rows >= 0
        if not keep.any():
            return empty
        cand_fids = cand_fids[keep]
        rows = rows[keep]

        C = feat[rows]                                   # (M, dim) candidate vectors
        Cconf = conf[rows]                               # (M, dim) candidate confidences
        qv = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        qc = np.asarray(query_conf, dtype=np.float32).reshape(-1)
        dim = int(self.dimension)

        qmask = qc >= min_confidence                     # (dim,)
        mask = qmask[None, :] & (Cconf >= min_confidence)  # (M, dim)
        valid_count = mask.sum(axis=1).astype(np.int64)   # (M,)

        diff = qv[None, :] - C                            # (M, dim)
        sq = diff * diff
        masked_sumsq = np.einsum('ij,ij->i', sq, mask.astype(np.float32))  # (M,)

        # Masked RMS-rescaled L2: ||diff_valid|| * sqrt(dim / valid_count). Mirrors
        # _compute_masked_distance exactly for the sufficient-overlap case.
        vc_safe = np.maximum(valid_count, 1)
        dist = np.sqrt(masked_sumsq) * np.sqrt(dim / vc_safe.astype(np.float32))

        # Last resort for candidates with ZERO mutually-valid dims: there is nothing to mask,
        # so fall back to the true (sqrt) full-vector L2. These are demoted below anyway.
        zero_valid = valid_count == 0
        if zero_valid.any():
            full_sumsq = sq.sum(axis=1)
            dist[zero_valid] = np.sqrt(full_sumsq[zero_valid])

        # Tier-1 demotion: any candidate below the overlap floor sorts after all tier-0 ones.
        fallback = valid_count < min_valid_overlap        # (M,) bool
        dist_final = dist.astype(np.float32, copy=True)
        dist_final[fallback] += self._FALLBACK_TIER_OFFSET

        order = np.argsort(dist_final, kind='stable')
        cand_fids = cand_fids[order]
        dist_final = dist_final[order]
        valid_count = valid_count[order]
        fallback = fallback[order]

        valid_dimensions_map: Dict[Any, int] = {}
        fallback_pose_ids: set = set()
        pose_id_map = self.pose_id_map
        for fid, vc, fb in zip(cand_fids.tolist(), valid_count.tolist(), fallback.tolist()):
            uuid_str = pose_id_map.get(fid)
            if uuid_str is None:
                continue
            pid = UUID(uuid_str)
            valid_dimensions_map[pid] = int(vc)
            if fb:
                fallback_pose_ids.add(pid)

        return cand_fids, dist_final, valid_dimensions_map, fallback_pose_ids

    def _reset_index_state(self) -> None:
        """Clear in-memory index state. Used after failed loads so close() won't re-save stale data."""
        self.index = None
        self.pose_id_map = {}
        self.uuid_to_faiss_id = {}
        self.next_id_counter = 0
        self._dirty_since_save = 0
        self._invalidate_corpus_cache()

    def _index_files_mtime(self):
        """(index mtime, mapping mtime) for staleness detection, or None."""
        try:
            return (self.index_path.stat().st_mtime, self.mapping_path.stat().st_mtime)
        except OSError:
            return None

    def reload_if_stale(self) -> bool:
        """Reload the index when another process has rebuilt it on disk.

        The app's resident search server loads the index once at launch; a
        completed re-detection (or any external rebuild) replaces the files
        and deletes the old pose rows — after which every search against the
        in-RAM index resolves to dead pose ids and returns nothing. One stat()
        per search keeps the server current without restarts.

        Returns True if a reload happened.
        """
        current = self._index_files_mtime()
        if current is None or current == getattr(self, '_loaded_index_mtime', None):
            return False
        logger.info("Index files changed on disk — reloading and clearing the search cache")
        with self._index_lock:
            # Re-check under the lock (another thread may have reloaded)
            current = self._index_files_mtime()
            if current == getattr(self, '_loaded_index_mtime', None):
                return False
            with self._cache_lock:
                self._search_cache.clear()
                self._search_cache_bytes = 0
            self._invalidate_corpus_cache()
            return self.load_index()

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

                        # Calibration scale (Wave 4); default when absent (legacy indices)
                        self.similarity_scale = float(
                            data_dict.get('similarity_scale', constants.DEFAULT_SIMILARITY_SCALE)
                        )

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
                        self._rebuild_uuid_reverse_map()
                        self._invalidate_corpus_cache()
                        self._dirty_since_save = 0
                        self._loaded_clean = True
                        self._loaded_index_mtime = self._index_files_mtime()
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
                logger.warning(f"Could not determine dimension from loaded index, "
                               f"using default {constants.GEOMETRIC_FEATURE_DIM}")
                self.dimension = constants.GEOMETRIC_FEATURE_DIM

            logger.info(f"Loaded index with dimension: {self.dimension}")

            self._rebuild_uuid_reverse_map()
            self._invalidate_corpus_cache()
            self._dirty_since_save = 0
            self._loaded_clean = True
            self._loaded_index_mtime = self._index_files_mtime()
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
        min_valid_overlap: int = 15,
        deduplicate_images: bool = True,
        required_regions: Optional[List[str]] = None,
        min_region_confidence: float = 0.3,
        min_similarity: float = 0.0,
        query_keypoints: Optional[np.ndarray] = None,
        query_bbox: Optional[np.ndarray] = None,
        exclude_pose_id: Optional[str] = None,
        query_visual_embedding: Optional[np.ndarray] = None
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
        # A re-detection or external rebuild replaces the index files and
        # deletes the old pose rows; without this check the resident search
        # server keeps serving an in-RAM index of dead ids (= zero results)
        # until the app restarts.
        self.reload_if_stale()

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
        # Visual rerank (finding 36) changes ordering/scores, so its state must key the cache:
        # the enable flag + weight, plus a short hash of the query visual embedding.
        from src.config.settings import settings as _settings
        visual_rerank_on = bool(_settings.ENABLE_VISUAL_RERANK) and query_visual_embedding is not None \
            and _settings.VISUAL_RERANK_WEIGHT > 0
        vis_hash = (
            hashlib.md5(np.asarray(query_visual_embedding, dtype=np.float32).tobytes()).hexdigest()[:8]
            if visual_rerank_on else 'none'
        )
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
            min_similarity,
            exclude_pose_id,
            vis_hash,
            round(_settings.VISUAL_RERANK_WEIGHT, 3) if visual_rerank_on else 0.0,
        )

        # Thread-safe cache check
        with self._cache_lock:
            if cache_key in self._search_cache:
                cached_results, timestamp, _approx_bytes = self._search_cache[cache_key]
                cache_age = time.time() - timestamp

                if cache_age < self._search_cache_ttl:
                    # Cache hit - move to end for LRU
                    self._search_cache.move_to_end(cache_key)
                    logger.debug(f"Search cache hit (age: {cache_age:.1f}s)")
                    # Shallow per-dict copy (finding 4): protects the cached list and its dicts
                    # from caller-side response shaping (paging, stripping heavy fields) without
                    # deep-copying immutable thumbnail bytes + nested keypoint lists on every hit,
                    # which was hundreds of ms to seconds for large sets. Callers must not mutate
                    # nested list VALUES (keypoints, regions) in place — they only ever reassign
                    # or delete dict keys, which a shallow copy already isolates.
                    return [dict(r) for r in cached_results]

                # Cache expired, will recompute
                self._evict_cache_entry(cache_key)
                logger.debug(f"Search cache expired ({cache_age:.1f}s)")

        # Phase 1: FAISS scan + confidence/OKS re-rank + base-similarity shortlist (no DB
        # hydration). Extracted so search_with_flip can score each leg cheaply and merge
        # before a single hydration pass (finding 5).
        candidates, valid_dimensions_map, oks_similarity_map = self._score_candidates(
            feature_vector,
            query_confidence=query_confidence,
            k=k,
            min_feature_confidence=min_feature_confidence,
            min_valid_overlap=min_valid_overlap,
            required_regions=required_regions,
            min_similarity=min_similarity,
            query_keypoints=query_keypoints,
            query_bbox=query_bbox,
        )

        # Optional visual rerank (finding 36): blend MobileNetV3 cosine into the top-N geometric
        # candidates before hydration. Gated OFF by default (ENABLE_VISUAL_RERANK / weight 0), so
        # the default path is untouched. Blended scores then flow through the normal floor + sort
        # in _hydrate_candidates, so enabling it makes the floor visual-aware by design.
        if visual_rerank_on:
            candidates = self._visual_rerank(
                candidates,
                query_visual_embedding,
                weight=_settings.VISUAL_RERANK_WEIGHT,
                top_n=_settings.VISUAL_RERANK_CANDIDATES,
            )

        # Phase 2/3: hydrate the shortlist once (shared with search_with_flip — finding 5).
        results = self._hydrate_candidates(
            candidates,
            k=k,
            min_confidence=min_confidence,
            deduplicate_images=deduplicate_images,
            required_regions=required_regions,
            min_region_confidence=min_region_confidence,
            min_similarity=min_similarity,
            exclude_pose_id=exclude_pose_id,
            valid_dimensions_map=valid_dimensions_map,
            oks_similarity_map=oks_similarity_map,
        )

        # Thread-safe cache store and eviction (count- AND byte-bounded — finding 4)
        with self._cache_lock:
            if cache_key in self._search_cache:
                # Overwrite: drop the old entry's byte contribution first.
                self._evict_cache_entry(cache_key)
            approx_bytes = self._estimate_results_bytes(results)
            self._search_cache[cache_key] = (results, time.time(), approx_bytes)
            self._search_cache_bytes += approx_bytes

            # Evict oldest entries until BOTH the count and byte budgets are satisfied. A single
            # large threshold-mode result set (thumbnails for thousands of poses) could otherwise
            # pin gigabytes; the byte bound caps total cache footprint regardless of query count.
            while self._search_cache and (
                len(self._search_cache) > self._search_cache_max_size
                or self._search_cache_bytes > self._search_cache_max_bytes
            ):
                oldest_key = next(iter(self._search_cache))
                if oldest_key == cache_key:
                    break  # never evict the entry we just inserted
                self._evict_cache_entry(oldest_key)
                logger.debug("Evicted oldest search cache entry (count/byte budget)")

        return results

    def _score_candidates(
        self,
        feature_vector: np.ndarray,
        query_confidence: Optional[np.ndarray],
        k: int,
        min_feature_confidence: float,
        min_valid_overlap: int,
        required_regions: Optional[List[str]],
        min_similarity: float,
        query_keypoints: Optional[np.ndarray],
        query_bbox: Optional[np.ndarray],
    ) -> Tuple[List[tuple], Dict, Dict]:
        """Phase 1: FAISS scan + confidence-aware/OKS re-rank + base-similarity shortlist.

        Returns (candidates, valid_dimensions_map, oks_similarity_map) WITHOUT touching the
        result cache or hydrating metadata/thumbnails. Extracted from search_by_feature so a
        flip search can score the direct and flipped legs cheaply, merge by pose, then hydrate
        once (finding 5).

        candidates is the ordered shortlist of (faiss_idx, distance, pose_id(UUID),
        base_similarity), sorted nearest-first. Returns an empty list for an absent/empty index
        or when the confidence re-rank yields nothing (caller then produces []); raises
        DimensionMismatchError on a dimension mismatch, exactly as before.
        """
        with self._index_lock:  # Thread-safe search
            if self.index is None:
                logger.error("Index not built. Call build_index() first.")
                return [], {}, {}

            if self.index.ntotal == 0:
                logger.warning("Index is empty")
                return [], {}, {}

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

        # === CONFIDENCE-AWARE RE-RANKING (vectorized, cached corpus) ===
        # valid_dimensions_map: pose_id(UUID) -> valid_count for the result dicts.
        # fallback_pose_ids: pose_ids re-ranked over <min_valid_overlap mutually-valid dims
        # (second-tier; sorted after every sufficient-overlap candidate, see finding 39).
        valid_dimensions_map = {}
        fallback_pose_ids: set = set()
        if query_confidence is not None:
            faiss_idx_array, distances_array, valid_dimensions_map, fallback_pose_ids = \
                self._rerank_confidence_aware(
                    indices[0],
                    distances[0],
                    np.asarray(feature_vector, dtype=np.float32).reshape(-1),
                    np.asarray(query_confidence, dtype=np.float32).reshape(-1),
                    min_feature_confidence,
                    min_valid_overlap,
                )
            if faiss_idx_array.size == 0:
                return [], {}, {}
            indices = faiss_idx_array.reshape(1, -1)
            distances = distances_array.reshape(1, -1)

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

        # Build the candidate shortlist.
        #
        # Phase 1 trims candidates cheaply by base similarity (derived from distance, no DB);
        # the caller's Phase 2/3 then bulk-fetches survivors. A candidate can only reach
        # final_similarity >= min_similarity if its base similarity (pre-plausibility) is
        # >= min_similarity / max_boost. plausibility_score <= 1.0, so the boost is bounded; use
        # that bound to stop scanning once distances grow too large (candidates are already
        # sorted nearest-first by L2 / masked / OKS distance).
        if settings.ENABLE_PLAUSIBILITY_SCORING:
            max_boost = 0.8 + 0.2 * settings.PLAUSIBILITY_WEIGHT
        else:
            max_boost = 1.0
        base_similarity_floor = (min_similarity / max_boost) if min_similarity > 0.0 else 0.0

        # In top-k mode (no threshold) bound how many candidates we bulk-load so we don't fetch
        # thumbnails for the entire re-ranking buffer; a small multiple of k covers filter
        # dropouts. In threshold mode the base-similarity early-stop bounds the shortlist instead.
        result_fetch_cap = None if min_similarity > 0.0 else max(k * 2, 200)

        # The floor early-stop assumes candidates are sorted similarity-descending. That holds
        # for the OKS / confidence-masked / plain paths (all distance-ascending). It does NOT
        # hold when finding-39 fallback candidates are present: they are demoted to a second
        # tier (sorted after every sufficient-overlap candidate) but their true masked distance
        # may be smaller than the tier boundary, so a tier-0 candidate dipping below the floor
        # does not imply the trailing tier-1 candidates are below it. Disable the break in that
        # case and floor-filter per candidate instead (still bounded by search_k / fetch cap).
        allow_early_stop = not fallback_pose_ids
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            dist_f = float(dist)
            if oks_active:
                # OKS distances are already 1-OKS in [0, 1]; mapping them through the L2
                # exp(-d/scale) curve would compress every score into [0.61, 1.0] and break
                # threshold semantics.
                base_sim = 1.0 - dist_f
            else:
                if query_confidence is None:
                    # Plain path: distances are RAW squared-L2 from IndexFlatL2. sqrt to true L2
                    # so _distance_to_similarity (calibrated for true L2) sees the same scale as
                    # the confidence-masked path, keeping the min_similarity floor consistent
                    # across modes (finding 2). Report the true-L2 value as 'distance' too, so the
                    # distance/base_similarity fields are comparable across modes. sqrt is
                    # monotonic, so the early-stop below stays valid.
                    dist_f = float(np.sqrt(max(dist_f, 0.0)))
                # else: confidence-masked path — distances are already true (RMS-rescaled) L2.
                base_sim = self._distance_to_similarity(dist_f)
            if min_similarity > 0.0 and base_sim < base_similarity_floor:
                if allow_early_stop:
                    # Sorted nearest-first → every remaining candidate is also below the floor.
                    break
                # Fallback tier present: skip this one, keep scanning for higher-sim tier-1 hits.
                continue
            candidates.append((int(idx), dist_f, UUID(self.pose_id_map[idx]), base_sim))
            if result_fetch_cap is not None and len(candidates) >= result_fetch_cap:
                break

        return candidates, valid_dimensions_map, oks_similarity_map

    def _visual_rerank(self, candidates, query_visual_embedding, weight: float, top_n: int):
        """Blend MobileNetV3 visual cosine into the top-N geometric candidates (finding 36).

        candidates: ordered (faiss_idx, distance, pose_id(UUID), base_similarity), geo-sorted.
        Only the top-N are reranked (the visual fetch is bounded by VISUAL_RERANK_CANDIDATES);
        the geometric tail keeps its order. Stored VisualFeatures vectors are L2-normalized, so
        cosine == dot; the query embedding is L2-normalized here to match. Blended similarity is
        (1-w)*geo + w*(cos+1)/2 (cosine mapped from [-1,1] to [0,1]); candidates lacking a stored
        embedding keep their geometric score. Returns a new candidate list with blended scores in
        the reranked prefix (re-sorted by blended score), followed by the untouched tail.
        """
        if not candidates or query_visual_embedding is None or weight <= 0:
            return candidates
        qv = np.asarray(query_visual_embedding, dtype=np.float32).reshape(-1)
        qnorm = float(np.linalg.norm(qv))
        if qnorm <= 0:
            return candidates
        qv = qv / qnorm

        n = min(int(top_n), len(candidates))
        head = candidates[:n]
        tail = candidates[n:]
        head_ids = [c[2] for c in head]

        from src.storage.models import VisualFeatures
        vis_map = {}
        with self.storage.session_scope() as session:
            for cs in range(0, len(head_ids), 1000):
                chunk = head_ids[cs:cs + 1000]
                for pid, vec in session.query(
                    VisualFeatures.pose_id, VisualFeatures.feature_vector
                ).filter(VisualFeatures.pose_id.in_(chunk)).all():
                    vis_map[pid] = np.asarray(vec, dtype=np.float32)

        reranked = []
        matched = 0
        for (idx, dist, pid, base_sim) in head:
            cv = vis_map.get(pid)
            if cv is not None and cv.shape == qv.shape:
                cn = float(np.linalg.norm(cv))
                cos = float(np.dot(qv, cv / cn)) if cn > 0 else 0.0
                cos01 = (cos + 1.0) / 2.0
                blended = (1.0 - weight) * base_sim + weight * cos01
                matched += 1
            else:
                blended = base_sim  # no stored visual embedding → geometric score unchanged
            reranked.append((idx, dist, pid, blended))

        reranked.sort(key=lambda c: c[3], reverse=True)
        logger.info(f"Visual rerank: blended {matched}/{len(head)} top candidates "
                    f"(w={weight:.2f}, {len(vis_map)} embeddings fetched)")
        return reranked + tail

    def _hydrate_candidates(
        self,
        candidates: List[tuple],
        k: int,
        min_confidence: float,
        deduplicate_images: bool,
        required_regions: Optional[List[str]],
        min_region_confidence: float,
        min_similarity: float,
        exclude_pose_id: Optional[str],
        valid_dimensions_map: Dict,
        oks_similarity_map: Dict,
        flip_flags: Optional[Dict[str, bool]] = None,
    ) -> List[Dict]:
        """Phase 2/3: bulk-fetch metadata + thumbnails for a candidate shortlist, filter,
        score, dedupe-by-image, and rank.

        Extracted from search_by_feature so search_with_flip can hydrate the MERGED
        direct+flipped shortlist exactly ONCE instead of running two full Phase 2/3 passes
        (two full thumbnail fetches) and discarding the overlap (finding 5).

        Args:
            candidates: ordered shortlist of (faiss_idx, distance, pose_id(UUID), base_similarity),
                sorted nearest-first. The k-cap during iteration assumes this ordering.
            flip_flags: optional {pose_id_str: is_flipped} — when provided, each result dict
                gets an 'is_flipped_match' key (flip-search merge semantics). Poses absent from
                the map default to False.

        Returns the ranked result list (best-first), identical in shape to search_by_feature's.
        """
        from src.config.settings import settings

        results: List[Dict] = []

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
                    # Exclude the query pose itself (search-by-pose-id) BEFORE the
                    # dedup check below, so its image's dedup slot goes to the
                    # next-best pose in that image instead of being consumed by the
                    # query pose and then filtered out downstream.
                    if exclude_pose_id is not None and str(pose_id) == exclude_pose_id:
                        continue
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

                    # Flip-search: mark which orientation won this pose (finding 5).
                    if flip_flags is not None:
                        result_dict['is_flipped_match'] = bool(flip_flags.get(str(pose_id), False))

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

        return results

    def _estimate_results_bytes(self, results: List[Dict]) -> int:
        """Approximate in-memory footprint of a cached result set. Thumbnail JPEG bytes
        dominate; a flat per-result overhead covers keypoints + metadata."""
        total = 0
        for r in results:
            thumb = r.get('thumbnail')
            if thumb:
                total += len(thumb)
            total += 2048  # keypoints (133x3) + metadata estimate
        return total

    def _evict_cache_entry(self, cache_key) -> None:
        """Remove one cache entry and decrement the running byte total. Caller holds _cache_lock."""
        entry = self._search_cache.pop(cache_key, None)
        if entry is not None and len(entry) >= 3:
            self._search_cache_bytes = max(0, self._search_cache_bytes - entry[2])

    def search_by_pose_id(
        self,
        pose_id: UUID,
        k: int = 20,
        exclude_self: bool = True,
        min_confidence: float = 0.0,
        min_feature_confidence: float = 0.35,
        min_valid_overlap: int = 15,
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
        # Visual rerank (finding 36): when enabled, use THIS pose's stored MobileNetV3 embedding
        # as the query visual so the "find similar to this indexed pose" flow reranks by
        # appearance with zero extra plumbing. No-op when the feature is off or no embedding exists.
        from src.config.settings import settings as _settings
        query_visual = None
        if _settings.ENABLE_VISUAL_RERANK and _settings.VISUAL_RERANK_WEIGHT > 0:
            from src.storage.models import VisualFeatures
            with self.storage.session_scope() as session:
                vf = session.query(VisualFeatures.feature_vector).filter(
                    VisualFeatures.pose_id == pose_id
                ).first()
                if vf is not None:
                    query_visual = np.asarray(vf[0], dtype=np.float32)

        search_k = k + 1 if exclude_self else k
        results = self.search_by_feature(
            query_vector,
            query_confidence=query_confidence,
            k=search_k,
            min_confidence=min_confidence,
            min_feature_confidence=min_feature_confidence,
            min_valid_overlap=min_valid_overlap,
            required_regions=required_regions,
            min_region_confidence=min_region_confidence,
            query_visual_embedding=query_visual
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
            self._search_cache_bytes = 0
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

        Uses inverse exponential decay: similarity = exp(-distance / scale).

        The scale comes from self.similarity_scale, which is loaded from the index
        metadata (finding 38) and fit by the re-extraction calibration step to the
        ACTUAL post-v3 distance distribution. The masked distance is already
        RMS-rescaled to the full-dimension equivalent, so no separate sqrt(dim)
        heuristic is applied here — the empirical calibration subsumes it. Default
        (constants.DEFAULT_SIMILARITY_SCALE) applies until a calibration is stored.
        """
        scale = self.similarity_scale if self.similarity_scale and self.similarity_scale > 0 \
            else constants.DEFAULT_SIMILARITY_SCALE

        similarity = float(np.exp(-distance / scale))

        # Log first result for debugging (avoid spam)
        if not hasattr(self, '_logged_similarity'):
            logger.info(f"Similarity calculation: distance={distance:.2f} → similarity={similarity:.2%} (dim={self.dimension}, scale={scale:.2f})")
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

        # Module-level sigma vectors (hoisted, finding 3): no per-call rebuild.
        extended_sigmas = _EXTENDED_SIGMAS

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

        # Vectorized OKS (finding 3): the old scalar 'for i in range(...)' loop over up to
        # 133 keypoints (with per-keypoint np.exp) is replaced by one masked array pass.
        # Numerically identical to the scalar version to ~1e-8 (only float summation order
        # differs); ranking and the inf/zero-valid fallbacks are preserved exactly.
        n = min(len(query_keypoints), len(candidate_keypoints), len(extended_sigmas))
        q = np.asarray(query_keypoints[:n], dtype=np.float32)
        c = np.asarray(candidate_keypoints[:n], dtype=np.float32)
        sig = extended_sigmas[:n]

        valid_mask = (q[:, 2] >= min_confidence) & (c[:, 2] >= min_confidence)
        valid_count = int(np.count_nonzero(valid_mask))

        if valid_count == 0:
            return (float('inf'), 0.0)

        # Distances in canonical (bbox-relative, scale-normalized) space.
        qx = (q[:, 0] - query_bbox[0]) / query_scale
        qy = (q[:, 1] - query_bbox[1]) / query_scale
        cx = (c[:, 0] - candidate_bbox[0]) / candidate_scale
        cy = (c[:, 1] - candidate_bbox[1]) / candidate_scale
        dx = qx - cx
        dy = qy - cy
        d_squared = dx * dx + dy * dy

        # OKS contribution: exp(-d²/(2κ²)) — s = 1 in canonical space. Only valid keypoints
        # contribute; masking the contributions to 0 reproduces the scalar loop's running sum.
        denominator = 2.0 * sig * sig
        oks_contributions = np.exp(-d_squared / denominator) * valid_mask
        oks_sum = float(oks_contributions.sum())

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

        # Each leg is SCORED (cheap Phase 1: FAISS scan + re-rank + base-similarity shortlist)
        # but NOT hydrated. The direct and flipped shortlists are merged by pose, then a SINGLE
        # Phase 2/3 pass fetches metadata + thumbnails for the merged survivors — instead of the
        # old design that ran the full pipeline (two full thumbnail fetches) twice and discarded
        # the heavy overlap (finding 5).
        #
        # In threshold mode (min_similarity > 0) the caller wants every match above the floor,
        # so each leg must score with the full k rather than the fixed merge-top-k.
        threshold_mode = search_kwargs.get('min_similarity', 0.0) > 0.0
        k_per_search = k if threshold_mode else settings.FLIP_SEARCH_MERGE_TOP_K

        # Args that _score_candidates accepts (Phase-1 only; hydration-time filters like
        # deduplicate_images / min_confidence / required-region presence are applied once in the
        # shared hydration pass, exactly as the unified search_by_feature path does).
        score_kwargs = dict(
            min_feature_confidence=search_kwargs.get('min_feature_confidence', 0.35),
            min_valid_overlap=search_kwargs.get('min_valid_overlap', 12),
            required_regions=search_kwargs.get('required_regions'),
            min_similarity=search_kwargs.get('min_similarity', 0.0),
            query_keypoints=None,  # OKS already stripped above
            query_bbox=None,
        )

        # Score the direct (normal) leg.
        direct_cands, direct_vdm, _direct_oks = self._score_candidates(
            feature_vector,
            query_confidence=query_confidence,
            k=k_per_search,
            **score_kwargs
        )

        # Flip keypoints and re-extract features
        flipped_keypoints = self._flip_keypoints_horizontal(keypoints)

        # Extract geometric features from flipped pose. The extractor takes a PoseResult,
        # not a raw array — derive visibility from keypoint confidence and a tight bbox
        # from the confident keypoints.
        from src.core.geometric_feature_extractor import GeometricFeatureExtractor
        from src.core.pose_detector import PoseResult

        # Visibility MUST be derived with the SAME confidence→COCO mapping the direct/stored
        # path uses (pose_detector.py: conf>=0.5 → 2 visible, 0.1<=conf<0.5 → 1 occluded,
        # conf<0.1 → 0 missing). The old np.where(conf>=0.3, 2, 0) forced every confident
        # keypoint to visible=2, giving it full trust in the extractor's linear vis-trust
        # curve (vis_trust = (vis/2)**1.0) and bypassing the halving that occluded (vis=1)
        # keypoints get on the direct/stored side — which skewed flipped vs direct scores
        # for occluded queries (finding 20). Mirroring the detector's mapping makes the
        # flipped pose's feature_confidence comparable to stored candidates'.
        flipped_conf = flipped_keypoints[:, 2]
        flipped_vis = np.zeros(len(flipped_keypoints), dtype=np.int64)
        flipped_vis[(flipped_conf >= 0.1) & (flipped_conf < 0.5)] = 1  # occluded
        flipped_vis[flipped_conf >= 0.5] = 2  # visible
        # Bbox is still derived from keypoints the extractor will actually use (vis >= 1.0,
        # i.e. confidence >= 0.1, matching the extractor's use_occluded_keypoints policy).
        confident = flipped_keypoints[flipped_conf >= 0.1]
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

        # Score the flipped leg. The confidence vector must be the flipped pose's own — the
        # extractor just computed it from the mirrored keypoints, so its left/right dims are
        # already swapped to match the flipped feature vector. Reusing the unflipped
        # query_confidence masked exactly the wrong side for asymmetrically occluded queries.
        flip_cands, flip_vdm, _flip_oks = self._score_candidates(
            flipped_features.feature_vector,
            query_confidence=(
                flipped_features.feature_confidence
                if query_confidence is not None else None
            ),
            k=k_per_search,
            **score_kwargs
        )

        # Merge the two shortlists by pose, keeping the better (smaller distance) orientation.
        # OKS is stripped for flip search, so both legs score on the same L2 / masked-L2 scale,
        # and the plausibility boost is per-pose and orientation-independent — so keeping the
        # smaller distance reproduces the old "keep the higher final similarity_score" merge
        # exactly, without hydrating both legs first.
        #
        # merged: pose_id_str -> (faiss_idx, distance, pose_id(UUID), base_similarity)
        merged: Dict[str, tuple] = {}
        flip_flags: Dict[str, bool] = {}
        merged_vdm: Dict[Any, int] = {}

        for cand in direct_cands:
            pid_str = str(cand[2])
            merged[pid_str] = cand
            flip_flags[pid_str] = False
            if cand[2] in direct_vdm:
                merged_vdm[cand[2]] = direct_vdm[cand[2]]

        for cand in flip_cands:
            pid_str = str(cand[2])
            existing = merged.get(pid_str)
            # Smaller distance (cand[1]) wins. Strict '<' so a tie keeps the direct leg,
            # matching the old strict '>' on similarity (direct inserted first, only replaced
            # when the flipped score was strictly greater).
            if existing is None or cand[1] < existing[1]:
                merged[pid_str] = cand
                flip_flags[pid_str] = True
                if cand[2] in flip_vdm:
                    merged_vdm[cand[2]] = flip_vdm[cand[2]]
                elif cand[2] in merged_vdm:
                    del merged_vdm[cand[2]]

        # Hydrate the merged shortlist ONCE, nearest-first so the top-k cap (non-threshold mode)
        # keeps the best across both orientations. The fallback-tier offset is baked into the
        # distances, so distance-ascending sorts sufficient-overlap candidates ahead of fallbacks.
        merged_candidates = sorted(merged.values(), key=lambda c: c[1])

        final_results = self._hydrate_candidates(
            merged_candidates,
            k=k,
            min_confidence=search_kwargs.get('min_confidence', 0.0),
            deduplicate_images=search_kwargs.get('deduplicate_images', True),
            required_regions=search_kwargs.get('required_regions'),
            min_region_confidence=search_kwargs.get('min_region_confidence', 0.3),
            min_similarity=search_kwargs.get('min_similarity', 0.0),
            exclude_pose_id=search_kwargs.get('exclude_pose_id'),
            valid_dimensions_map=merged_vdm,
            oks_similarity_map={},
            flip_flags=flip_flags,
        )

        logger.info(
            f"Flip search: {len(direct_cands)} direct + {len(flip_cands)} flipped candidates "
            f"→ {len(merged_candidates)} merged → {len(final_results)} hydrated"
        )

        return final_results
