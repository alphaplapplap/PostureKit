"""
Storage Manager for PostureKit.
Coordinates atomic writes across all database tables.
"""
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path
from uuid import UUID
from datetime import datetime
import logging
import numpy as np
import weakref
import threading

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from src.config.settings import settings
from src.storage.models import (
    Base, Image, PoseDetection, GeometricFeatures,
    VisualFeatures, FusedFeatures, TrainingLabel, CorrectionStatistics, CorrectionEvent
)
from src.core.pose_detector import PoseResult
from src.core.geometric_feature_extractor import GeometricFeatures as GeometricFeaturesData
from src.core.visual_feature_extractor import VisualFeatures as VisualFeaturesData
from src.core.multimodal_fusion import FusedFeatures as FusedFeaturesData
from src.core.image_ingestor import ImageMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StorageManagerError(Exception):
    """Base exception for Storage Manager errors."""
    pass


class StorageManager:
    """
    Manages database operations for PostureKit.

    Features:
    - Atomic transactions across multiple tables
    - Shared connection pool across all instances (prevents pool exhaustion)
    - Thread-local sessions for thread safety
    - Automatic rollback on errors
    - Context manager for sessions

    Attributes:
        engine: SQLAlchemy engine (shared across instances)
        SessionLocal: Thread-local scoped session factory
    """

    # Class-level shared resources to prevent pool exhaustion
    _engines = {}  # connection_string -> engine mapping
    _engine_lock = threading.Lock()  # Protects engine creation

    def __init__(self, connection_string: Optional[str] = None, database_profile: Optional[str] = None):
        """
        Initialize Storage Manager with shared connection pool.

        Args:
            connection_string: PostgreSQL connection string (uses settings default if None)
            database_profile: Database profile (irl/2d/3d). Overrides connection_string if provided.
        """
        # Database profile takes precedence over connection_string
        if database_profile:
            self.connection_string = settings.get_database_url(database_profile)
            self.database_profile = database_profile
        else:
            self.connection_string = connection_string or settings.DATABASE_URL
            self.database_profile = settings.DB_PROFILE

        # Get or create shared engine (prevents pool exhaustion)
        with self._engine_lock:
            if self.connection_string not in self._engines:
                # Dynamic pool sizing based on workload
                import os
                cpu_count = os.cpu_count() or 4
                pool_size = max(20, cpu_count * 2)  # Minimum 20, scale with CPUs
                max_overflow = pool_size * 2  # Allow 2x burst capacity

                # Create shared engine with improved connection pooling
                self._engines[self.connection_string] = create_engine(
                    self.connection_string,
                    pool_size=pool_size,  # Shared across all StorageManager instances
                    max_overflow=max_overflow,  # Shared overflow capacity
                    pool_timeout=60,  # Longer timeout for batch workloads
                    pool_recycle=3600,  # Recycle connections after 1 hour
                    pool_pre_ping=True,  # Verify connections before using
                    poolclass=QueuePool,
                    pool_reset_on_return='rollback',  # Reset connection state
                    echo=False  # Set to True for SQL debugging
                )
                logger.info(f"Database pool: size={pool_size}, max_overflow={max_overflow}")
                logger.info(
                    f"Created shared database engine for {self.connection_string.split('@')[1] if '@' in self.connection_string else 'unknown'}"
                )

            self.engine = self._engines[self.connection_string]

        # Create thread-local scoped session factory
        # Each thread gets its own session, preventing concurrent access issues
        session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        self.SessionLocal = scoped_session(session_factory)

        # Track active sessions with weak references (thread-safe)
        self._active_sessions = weakref.WeakSet()
        self._session_lock = threading.RLock()  # Protects session tracking

        logger.info(
            "StorageManager initialized with shared pool",
            extra={'extra_data': {
                'connection_string': self.connection_string.split('@')[1] if '@' in self.connection_string else 'unknown',
                'shared_pool_size': 10,
                'shared_max_overflow': 20,
                'thread_local_sessions': True
            }}
        )
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with storage_manager.session_scope() as session:
                # Do database operations
                pass
        """
        session = self.SessionLocal()

        # Thread-safe session tracking
        with self._session_lock:
            self._active_sessions.add(session)

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session rollback due to error: {e}", exc_info=True)
            raise
        finally:
            # Explicitly expunge all objects to break circular references
            try:
                session.expunge_all()
            finally:
                session.close()
                # Connection automatically returned to pool
    
    def store_detection(
        self,
        image_path: Path,
        image_metadata: ImageMetadata,
        pose_result: PoseResult,
        features: GeometricFeaturesData,
        visual_features: Optional[VisualFeaturesData] = None,
        fused_features: Optional[FusedFeaturesData] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_notes: Optional[str] = None,
        thumbnail_bytes: Optional[bytes] = None,
        body_part_detections: Optional[List] = None
    ) -> UUID:
        """
        Store complete detection results atomically.

        Args:
            image_path: Path to image file
            image_metadata: Metadata from image ingestor
            pose_result: Detected pose from pose detector
            features: Extracted geometric features
            visual_features: Optional VisualFeatures from visual_feature_extractor
            fused_features: Optional FusedFeatures from multimodal fusion
            category: Optional pose category
            difficulty: Optional annotation difficulty
            tags: Optional list of tags
            user_notes: Optional user notes
            thumbnail_bytes: Optional pre-generated thumbnail as JPEG bytes
            body_part_detections: Optional list of BodyPartDetection objects from NudeNet

        Returns:
            UUID of created pose_detection record

        Raises:
            StorageManagerError: If storage fails
        """
        logger.info(f"Storing detection for {image_path.name}")

        try:
            with self.session_scope() as session:
                # 1. Create or get Image record
                image = session.execute(
                    select(Image).where(Image.file_path == str(image_path))
                ).scalar_one_or_none()

                if image is None:
                    image = Image(
                        file_path=str(image_path),
                        content_hash=image_metadata.content_hash,
                        width=image_metadata.original_width,
                        height=image_metadata.original_height,
                        file_size_bytes=image_metadata.file_size_bytes,
                        thumbnail=thumbnail_bytes
                    )
                    session.add(image)
                    session.flush()  # Get image.id
                    logger.debug(f"Created Image record: {image.id} (thumbnail: {len(thumbnail_bytes) if thumbnail_bytes else 0} bytes)")
                else:
                    # Update thumbnail if provided and not already set
                    if thumbnail_bytes and not image.thumbnail:
                        image.thumbnail = thumbnail_bytes
                        logger.debug(f"Updated thumbnail for existing Image: {image.id}")
                    logger.debug(f"Using existing Image record: {image.id}")

                # 2. Create PoseDetection record
                pose_detection = PoseDetection(
                    image_id=image.id,
                    bbox=pose_result.bbox.tolist(),
                    overall_confidence=float(pose_result.overall_confidence),
                    person_id=pose_result.person_id,
                    is_corrected=False
                )
                # Set keypoints and visibility using helper methods
                pose_detection.set_keypoints(pose_result.keypoints)
                pose_detection.set_visibility(pose_result.visibility)

                session.add(pose_detection)
                session.flush()  # Get pose_detection.id
                logger.debug(f"Created PoseDetection record: {pose_detection.id}")

                # 3. Create GeometricFeatures record
                geometric_features_rec = GeometricFeatures(
                    pose_id=pose_detection.id,
                    feature_vector=features.feature_vector.tolist(),
                    feature_confidence=features.feature_confidence.tolist(),
                    joint_angles=features.joint_angles,
                    limb_ratios=features.limb_ratios,
                    body_angles=features.body_angles,
                    symmetry_scores=features.symmetry_scores,
                    occlusion_pattern=features.occlusion_pattern.tolist()
                )
                session.add(geometric_features_rec)
                session.flush()  # Get geometric_features_rec.id
                logger.debug(f"Created GeometricFeatures record: {geometric_features_rec.id}")

                # 4. Create VisualFeatures record if provided
                visual_features_rec = None
                logger.debug(f"Visual features provided: {visual_features is not None}")
                if visual_features is not None:
                    visual_features_rec = VisualFeatures(
                        pose_id=pose_detection.id,
                        feature_vector=visual_features.feature_vector.tolist(),
                        model_name=visual_features.model_name,
                        normalization=visual_features.normalization
                    )
                    session.add(visual_features_rec)
                    session.flush()  # Get visual_features_rec.id
                    logger.debug(f"Created VisualFeatures record: {visual_features_rec.id}")
                else:
                    logger.warning(f"Visual features NOT provided for pose {pose_detection.id}, skipping VisualFeatures record")

                # 5. Create FusedFeatures record if provided
                logger.debug(f"FusedFeatures creation check: fused_features={'provided' if fused_features is not None else 'None'}, visual_features_rec={'exists' if visual_features_rec is not None else 'None'}")
                if fused_features is not None and visual_features_rec is not None:
                    fused_features_rec = FusedFeatures(
                        pose_id=pose_detection.id,
                        geometric_feature_id=geometric_features_rec.id,
                        visual_feature_id=visual_features_rec.id,
                        fused_vector=fused_features.fused_vector.tolist(),
                        geometric_vector=fused_features.geometric_vector.tolist(),
                        visual_vector=fused_features.visual_vector.tolist(),
                        fusion_method=fused_features.fusion_method
                    )
                    session.add(fused_features_rec)
                    logger.debug(f"Created FusedFeatures record: {fused_features_rec.id}")
                else:
                    logger.warning(
                        f"FusedFeatures NOT created for pose {pose_detection.id} - "
                        f"fused_features={'provided' if fused_features is not None else 'MISSING'}, "
                        f"visual_features_rec={'exists' if visual_features_rec is not None else 'MISSING'}"
                    )

                # 6. Create TrainingLabel record if any labels provided
                if category or difficulty or tags or user_notes:
                    training_label = TrainingLabel(
                        pose_id=pose_detection.id,
                        category=category,
                        difficulty=difficulty,
                        tags=tags,
                        user_notes=user_notes
                    )
                    session.add(training_label)
                    logger.debug(f"Created TrainingLabel record: {training_label.id}")

                # 7. Create BodyPart records if provided
                body_parts_stored = 0
                if body_part_detections:
                    from src.storage.models import BodyPart
                    for detection in body_part_detections:
                        body_part = BodyPart(
                            image_id=image.id,
                            person_index=pose_result.person_id,
                            part_name=detection.part_name,
                            canonical_region=detection.canonical_region,
                            confidence=float(detection.confidence),
                            bbox=detection.bbox.tolist(),
                            is_exposed=detection.is_exposed
                        )
                        session.add(body_part)
                        body_parts_stored += 1
                    logger.debug(f"Created {body_parts_stored} BodyPart records for image {image.id}")

                # Commit happens in context manager
                has_fused_record = fused_features is not None and visual_features_rec is not None
                logger.info(
                    f"Stored detection successfully",
                    extra={'extra_data': {
                        'pose_id': str(pose_detection.id),
                        'image_id': str(image.id),
                        'visible_keypoints': pose_result.count_visible(),
                        'has_visual': visual_features is not None,
                        'has_visual_record': visual_features_rec is not None,
                        'has_fused': fused_features is not None,
                        'has_fused_record': has_fused_record
                    }}
                )

                if not has_fused_record:
                    logger.warning(f"Pose {pose_detection.id} stored WITHOUT FusedFeatures record - similarity search will fail!")

                return pose_detection.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to store detection: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e

    def store_multi_person_detections(
        self,
        image_path: Path,
        image_metadata: ImageMetadata,
        pose_results: List[PoseResult],
        features_list: List[GeometricFeaturesData],
        visual_features_list: Optional[List[VisualFeaturesData]] = None,
        fused_features_list: Optional[List[FusedFeaturesData]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_notes: Optional[str] = None,
        thumbnail_bytes: Optional[bytes] = None,
        body_part_detections_list: Optional[List[List]] = None
    ) -> List[UUID]:
        """
        Store multiple person detections from a single image atomically.

        All detections are stored in a single transaction for consistency.
        If any detection fails, all are rolled back.

        Args:
            image_path: Path to image file
            image_metadata: Metadata from image ingestor
            pose_results: List of detected poses (one per person)
            features_list: List of geometric features (one per person)
            visual_features_list: Optional list of visual features (one per person)
            fused_features_list: Optional list of fused features (one per person)
            category: Optional pose category (applies to all persons)
            difficulty: Optional annotation difficulty (applies to all persons)
            tags: Optional list of tags (applies to all persons)
            user_notes: Optional user notes (applies to all persons)
            thumbnail_bytes: Optional pre-generated thumbnail (shared by all persons)
            body_part_detections_list: Optional list of body part detections per person

        Returns:
            List of UUIDs for created pose_detection records

        Raises:
            StorageManagerError: If storage fails
            ValueError: If list lengths don't match
        """
        num_persons = len(pose_results)

        # Validate list lengths
        if len(features_list) != num_persons:
            raise ValueError(f"features_list length ({len(features_list)}) must match pose_results ({num_persons})")

        if visual_features_list is not None and len(visual_features_list) != num_persons:
            raise ValueError(f"visual_features_list length ({len(visual_features_list)}) must match pose_results ({num_persons})")

        if fused_features_list is not None and len(fused_features_list) != num_persons:
            raise ValueError(f"fused_features_list length ({len(fused_features_list)}) must match pose_results ({num_persons})")

        if body_part_detections_list is not None and len(body_part_detections_list) != num_persons:
            raise ValueError(f"body_part_detections_list length ({len(body_part_detections_list)}) must match pose_results ({num_persons})")

        logger.info(f"Storing {num_persons} person detection(s) for {image_path.name}")

        try:
            pose_ids = []

            # Store all persons in a single transaction
            with self.session_scope() as session:
                # 1. Create or get Image record (shared by all persons)
                image = session.execute(
                    select(Image).where(Image.file_path == str(image_path))
                ).scalar_one_or_none()

                if image is None:
                    image = Image(
                        file_path=str(image_path),
                        content_hash=image_metadata.content_hash,
                        width=image_metadata.original_width,
                        height=image_metadata.original_height,
                        file_size_bytes=image_metadata.file_size_bytes,
                        thumbnail=thumbnail_bytes
                    )
                    session.add(image)
                    session.flush()  # Get image.id
                    logger.debug(f"Created Image record: {image.id}")
                else:
                    # Update thumbnail if provided and not already set
                    if thumbnail_bytes and not image.thumbnail:
                        image.thumbnail = thumbnail_bytes
                    logger.debug(f"Using existing Image record: {image.id}")

                # 2. Store each person's detection
                for person_idx in range(num_persons):
                    pose_result = pose_results[person_idx]
                    features = features_list[person_idx]
                    visual_features = visual_features_list[person_idx] if visual_features_list else None
                    fused_features = fused_features_list[person_idx] if fused_features_list else None
                    body_part_detections = body_part_detections_list[person_idx] if body_part_detections_list else None

                    logger.debug(f"Storing person {person_idx + 1}/{num_persons} (person_id={pose_result.person_id})")

                    # Create PoseDetection record
                    pose_detection = PoseDetection(
                        image_id=image.id,
                        bbox=pose_result.bbox.tolist(),
                        overall_confidence=float(pose_result.overall_confidence),
                        person_id=pose_result.person_id,
                        is_corrected=False
                    )
                    pose_detection.set_keypoints(pose_result.keypoints)
                    pose_detection.set_visibility(pose_result.visibility)

                    session.add(pose_detection)
                    session.flush()  # Get pose_detection.id
                    pose_ids.append(pose_detection.id)

                    # Create GeometricFeatures record
                    geometric_features_rec = GeometricFeatures(
                        pose_id=pose_detection.id,
                        feature_vector=features.feature_vector.tolist(),
                        feature_confidence=features.feature_confidence.tolist(),
                        joint_angles=features.joint_angles,
                        limb_ratios=features.limb_ratios,
                        body_angles=features.body_angles,
                        symmetry_scores=features.symmetry_scores,
                        occlusion_pattern=features.occlusion_pattern.tolist()
                    )
                    session.add(geometric_features_rec)
                    session.flush()

                    # Create VisualFeatures record if provided
                    visual_features_rec = None
                    if visual_features is not None:
                        visual_features_rec = VisualFeatures(
                            pose_id=pose_detection.id,
                            feature_vector=visual_features.feature_vector.tolist(),
                            model_name=visual_features.model_name,
                            normalization=visual_features.normalization
                        )
                        session.add(visual_features_rec)
                        session.flush()

                    # Create FusedFeatures record if provided
                    if fused_features is not None and visual_features_rec is not None:
                        fused_features_rec = FusedFeatures(
                            pose_id=pose_detection.id,
                            geometric_feature_id=geometric_features_rec.id,
                            visual_feature_id=visual_features_rec.id,
                            fused_vector=fused_features.fused_vector.tolist(),
                            geometric_vector=fused_features.geometric_vector.tolist(),
                            visual_vector=fused_features.visual_vector.tolist(),
                            fusion_method=fused_features.fusion_method
                        )
                        session.add(fused_features_rec)

                    # Create TrainingLabel record if any labels provided
                    if category or difficulty or tags or user_notes:
                        training_label = TrainingLabel(
                            pose_id=pose_detection.id,
                            category=category,
                            difficulty=difficulty,
                            tags=tags,
                            user_notes=user_notes
                        )
                        session.add(training_label)

                    # Create BodyPart records if provided
                    if body_part_detections:
                        from src.storage.models import BodyPart
                        for detection in body_part_detections:
                            body_part = BodyPart(
                                image_id=image.id,
                                person_index=pose_result.person_id,
                                part_name=detection.part_name,
                                canonical_region=detection.canonical_region,
                                confidence=float(detection.confidence),
                                bbox=detection.bbox.tolist(),
                                is_exposed=detection.is_exposed
                            )
                            session.add(body_part)

                # Commit happens in context manager
                logger.info(
                    f"Stored {num_persons} person detection(s) successfully",
                    extra={'extra_data': {
                        'image_id': str(image.id),
                        'pose_ids': [str(pid) for pid in pose_ids],
                        'num_persons': num_persons
                    }}
                )

                return pose_ids

        except SQLAlchemyError as e:
            logger.error(f"Failed to store multi-person detections: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e

    def store_correction_statistics(
        self,
        pose_id: UUID,
        original_keypoints: np.ndarray,
        corrected_keypoints: np.ndarray,
        session = None
    ) -> UUID:
        """
        Store correction statistics for a pose.

        Args:
            pose_id: ID of the pose that was corrected
            original_keypoints: Original keypoints (133, 3)
            corrected_keypoints: Corrected keypoints (133, 3)
            session: Optional existing session (for transactions)

        Returns:
            ID of created CorrectionStatistics record
        """
        from src.learning.correction_learner import CorrectionLearner

        # Compute statistics
        stats = CorrectionLearner.compute_correction_statistics(
            original_keypoints,
            corrected_keypoints
        )

        # Create record
        correction_stats = CorrectionStatistics(
            pose_id=pose_id,
            keypoints_corrected_count=stats['keypoints_corrected_count'],
            avg_correction_distance=stats['avg_correction_distance'],
            max_correction_distance=stats['max_correction_distance'],
            corrected_keypoint_types=stats['corrected_keypoint_types'],
            avg_confidence_before=stats['avg_confidence_before'],
            avg_confidence_after=stats['avg_confidence_after']
        )

        # Store
        if session:
            session.add(correction_stats)
            session.flush()
        else:
            with self.session_scope() as sess:
                sess.add(correction_stats)
                sess.flush()
                stats_id = correction_stats.id
            return stats_id

        return correction_stats.id

    def _build_pose_dict(self, pose: PoseDetection, session: Session) -> Dict[str, Any]:
        """
        Build pose dictionary from ORM object within session context.

        Args:
            pose: PoseDetection ORM object
            session: Active SQLAlchemy session

        Returns:
            Dictionary with all related data
        """
        # Build result dictionary - access all relationships while session is active
        result = {
            'pose_id': pose.id,
            'image_id': pose.image_id,
            'image_path': pose.image.file_path,
            'keypoints': pose.keypoints,
            'keypoint_visibility': pose.keypoint_visibility,
            'bbox': pose.bbox,
            'confidence': pose.overall_confidence,
            'person_id': pose.person_id,
            'is_corrected': pose.is_corrected,
            'created_at': pose.created_at
        }

        # Add geometric features if available
        if pose.geometric_features:
            result['geometric_features'] = {
                'feature_vector': pose.geometric_features.feature_vector,
                'joint_angles': pose.geometric_features.joint_angles,
                'limb_ratios': pose.geometric_features.limb_ratios,
                'body_angles': pose.geometric_features.body_angles,
                'symmetry_scores': pose.geometric_features.symmetry_scores,
                'occlusion_pattern': pose.geometric_features.occlusion_pattern
            }

        # Add visual features if available
        if pose.visual_features:
            result['visual_features'] = {
                'feature_vector': pose.visual_features.feature_vector,
                'model_name': pose.visual_features.model_name,
                'normalization': pose.visual_features.normalization
            }

        # Add fused features if available
        if pose.fused_features:
            result['fused_features'] = {
                'fused_vector': pose.fused_features.fused_vector,
                'geometric_vector': pose.fused_features.geometric_vector,
                'visual_vector': pose.fused_features.visual_vector,
                'fusion_method': pose.fused_features.fusion_method
            }

        # Add labels if available
        if pose.training_label:
            result['labels'] = {
                'category': pose.training_label.category,
                'difficulty': pose.training_label.difficulty,
                'tags': pose.training_label.tags,
                'user_notes': pose.training_label.user_notes
            }

        return result

    def get_pose_by_id(self, pose_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete pose data by ID.

        Args:
            pose_id: UUID of pose detection

        Returns:
            Dictionary with all related data, or None if not found
        """
        try:
            with self.session_scope() as session:
                pose = session.execute(
                    select(PoseDetection).where(PoseDetection.id == pose_id)
                ).scalar_one_or_none()

                if pose is None:
                    return None

                return self._build_pose_dict(pose, session)

        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve pose {pose_id}: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e
    
    def get_poses_by_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Retrieve all poses for a specific image.

        Args:
            image_path: Path to image file

        Returns:
            List of pose dictionaries
        """
        try:
            with self.session_scope() as session:
                poses = session.execute(
                    select(PoseDetection)
                    .join(Image)
                    .where(Image.file_path == str(image_path))
                    .order_by(PoseDetection.person_id)
                ).scalars().all()

                # Build results within same session - FIXES N+1 query problem
                return [self._build_pose_dict(pose, session) for pose in poses]

        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve poses for {image_path}: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e
    
    def update_detection(
        self,
        pose_id: UUID,
        corrected_keypoints: np.ndarray,
        corrected_visibility: np.ndarray,
        corrected_features: GeometricFeaturesData,
        correction_type: Optional[str] = None,
        store_statistics: bool = True
    ) -> bool:
        """
        Update existing pose detection with corrections and re-extracted features.

        Args:
            pose_id: UUID of pose detection to update
            corrected_keypoints: Corrected keypoints (133, 3) array
            corrected_visibility: Corrected visibility (133,) array
            corrected_features: Re-extracted geometric features
            correction_type: Type of correction ('position', 'occlusion', 'both')
            store_statistics: Whether to store correction statistics

        Returns:
            True if successful

        Raises:
            StorageManagerError: If update fails
        """
        logger.info(f"Updating detection for pose {pose_id}")

        try:
            with self.session_scope() as session:
                # Get original pose
                pose = session.execute(
                    select(PoseDetection).where(PoseDetection.id == pose_id)
                ).scalar_one_or_none()

                if pose is None:
                    logger.warning(f"Pose {pose_id} not found")
                    return False

                # Store original keypoints for statistics if needed
                original_keypoints = pose.get_keypoints_array() if store_statistics else None

                # Update pose detection
                pose.set_keypoints(corrected_keypoints)
                pose.set_visibility(corrected_visibility)
                pose.is_corrected = True
                if correction_type:
                    pose.correction_type = correction_type

                # Update geometric features
                geom_features = session.execute(
                    select(GeometricFeatures).where(GeometricFeatures.pose_id == pose_id)
                ).scalar_one_or_none()

                if geom_features:
                    geom_features.feature_vector = corrected_features.feature_vector.tolist()
                    geom_features.joint_angles = corrected_features.joint_angles
                    geom_features.limb_ratios = corrected_features.limb_ratios
                    geom_features.body_angles = corrected_features.body_angles
                    geom_features.symmetry_scores = corrected_features.symmetry_scores
                    geom_features.occlusion_pattern = corrected_features.occlusion_pattern.tolist()

                # Store correction statistics if requested
                if store_statistics and original_keypoints is not None:
                    self.store_correction_statistics(
                        pose_id=pose_id,
                        original_keypoints=original_keypoints,
                        corrected_keypoints=corrected_keypoints,
                        session=session
                    )

                logger.info(
                    f"Updated detection successfully",
                    extra={'extra_data': {
                        'pose_id': str(pose_id),
                        'correction_type': correction_type,
                        'statistics_stored': store_statistics
                    }}
                )
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to update detection {pose_id}: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e

    def save_pose_correction(
        self,
        pose_id: UUID,
        corrected_keypoints: np.ndarray,
        corrected_visibility: Optional[np.ndarray] = None
    ) -> None:
        """
        Save manual correction to a pose.

        On first correction, freezes original keypoints. On subsequent corrections,
        updates current keypoints and logs the change.

        Args:
            pose_id: UUID of pose to correct
            corrected_keypoints: New keypoint positions (133, 3) or (399,)
            corrected_visibility: Optional visibility flags (133,)
        """
        with self.session_scope() as session:
            # Use SELECT FOR UPDATE to prevent concurrent modifications
            from sqlalchemy import select
            stmt = select(PoseDetection).where(
                PoseDetection.id == pose_id
            ).with_for_update()
            pose = session.execute(stmt).scalar_one_or_none()

            if not pose:
                raise ValueError(f"Pose {pose_id} not found")

            # Reshape if needed
            if corrected_keypoints.shape == (133, 3):
                corrected_keypoints = corrected_keypoints.flatten()

            # Validate corrected_keypoints size before reshape
            if corrected_keypoints.size != 399:
                raise ValueError(
                    f"Corrected keypoints must have 399 elements (133 * 3), got {corrected_keypoints.size}"
                )

            # Convert current keypoints to array for comparison with validation
            current_kp_array = np.array(pose.keypoints, dtype=np.float32)
            if current_kp_array.size != 399:
                raise ValueError(
                    f"Pose {pose_id}: Corrupted keypoints in database (expected 399, got {current_kp_array.size})"
                )

            current_keypoints = current_kp_array.reshape(133, 3)
            new_keypoints = corrected_keypoints.reshape(133, 3)

            # Calculate what changed
            deltas = new_keypoints[:, :2] - current_keypoints[:, :2]  # Only XY, not confidence
            displacements = np.linalg.norm(deltas, axis=1)
            changed_mask = displacements > 1.0  # Moved more than 1 pixel
            changed_indices = np.where(changed_mask)[0]

            # First correction? Freeze original state
            if not pose.is_corrected:
                pose.original_keypoints = pose.keypoints.copy()
                if pose.keypoint_visibility:
                    pose.original_keypoint_visibility = pose.keypoint_visibility.copy()
                pose.first_corrected_at = datetime.utcnow()
                pose.is_corrected = True
                logger.info(f"First correction for pose {pose_id}, froze original keypoints")

            # Update current state
            pose.keypoints = corrected_keypoints.tolist()
            if corrected_visibility is not None:
                pose.keypoint_visibility = corrected_visibility.tolist()
            pose.correction_count += 1
            pose.updated_at = datetime.utcnow()

            # Log this correction event
            if len(changed_indices) > 0:
                event = CorrectionEvent(
                    pose_id=pose_id,
                    correction_number=pose.correction_count,
                    changed_keypoint_indices=changed_indices.tolist(),
                    max_displacement=float(displacements[changed_mask].max()),
                    avg_displacement=float(displacements[changed_mask].mean())
                )
                session.add(event)

                logger.info(
                    f"Saved correction {pose.correction_count} for pose {pose_id}",
                    extra={'extra_data': {
                        'changed_keypoints': len(changed_indices),
                        'max_displacement': float(displacements[changed_mask].max())
                    }}
                )

    def update_pose_correction(
        self,
        pose_id: UUID,
        keypoints: Optional[np.ndarray] = None,
        visibility: Optional[np.ndarray] = None,
        correction_type: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_notes: Optional[str] = None
    ) -> bool:
        """
        Update pose with user corrections.

        Args:
            pose_id: UUID of pose detection
            keypoints: Updated keypoints (133, 3) array if corrected
            visibility: Updated visibility (133,) array if corrected
            correction_type: Type of correction ('position', 'occlusion', 'both')
            category: Updated category
            difficulty: Updated difficulty
            tags: Updated tags
            user_notes: Updated notes

        Returns:
            True if successful
        """
        logger.info(f"Updating pose correction for {pose_id}")

        try:
            with self.session_scope() as session:
                # Get pose
                pose = session.execute(
                    select(PoseDetection).where(PoseDetection.id == pose_id)
                ).scalar_one_or_none()

                if pose is None:
                    logger.warning(f"Pose {pose_id} not found")
                    return False

                # Update keypoints if provided
                if keypoints is not None:
                    pose.set_keypoints(keypoints)
                    pose.is_corrected = True

                # Update visibility if provided
                if visibility is not None:
                    pose.set_visibility(visibility)
                    pose.is_corrected = True

                # Update correction type
                if correction_type is not None:
                    pose.correction_type = correction_type

                # Update or create training label
                if category or difficulty or tags or user_notes:
                    if pose.training_label is None:
                        pose.training_label = TrainingLabel(pose_id=pose_id)

                    if category is not None:
                        pose.training_label.category = category
                    if difficulty is not None:
                        pose.training_label.difficulty = difficulty
                    if tags is not None:
                        pose.training_label.tags = tags
                    if user_notes is not None:
                        pose.training_label.user_notes = user_notes

                logger.info(f"Updated pose {pose_id} with corrections")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to update pose {pose_id}: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e
    
    def delete_pose(self, pose_id: UUID) -> bool:
        """
        Delete pose and all related records.
        
        Args:
            pose_id: UUID of pose detection
            
        Returns:
            True if deleted, False if not found
        """
        logger.info(f"Deleting pose {pose_id}")
        
        try:
            with self.session_scope() as session:
                pose = session.execute(
                    select(PoseDetection).where(PoseDetection.id == pose_id)
                ).scalar_one_or_none()
                
                if pose is None:
                    logger.warning(f"Pose {pose_id} not found")
                    return False
                
                session.delete(pose)  # CASCADE will delete related records
                logger.info(f"Deleted pose {pose_id}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete pose {pose_id}: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with counts and metrics
        """
        try:
            with self.session_scope() as session:
                total_images = session.execute(
                    select(func.count()).select_from(Image)
                ).scalar()

                total_poses = session.execute(
                    select(func.count()).select_from(PoseDetection)
                ).scalar()

                corrected_poses = session.execute(
                    select(func.count()).select_from(PoseDetection).where(PoseDetection.is_corrected == True)
                ).scalar()

                labeled_poses = session.execute(
                    select(func.count()).select_from(TrainingLabel)
                ).scalar()

                poses_with_visual = session.execute(
                    select(func.count()).select_from(VisualFeatures)
                ).scalar()

                poses_with_fused = session.execute(
                    select(func.count()).select_from(FusedFeatures)
                ).scalar()

                avg_confidence = session.execute(
                    select(func.avg(PoseDetection.overall_confidence))
                ).scalar()

                return {
                    'total_images': total_images,
                    'total_poses': total_poses,
                    'corrected_poses': corrected_poses,
                    'labeled_poses': labeled_poses,
                    'poses_with_visual_features': poses_with_visual,
                    'poses_with_fused_features': poses_with_fused,
                    'avg_confidence': float(avg_confidence) if avg_confidence else 0.0,
                    'visual_feature_coverage': poses_with_visual / total_poses if total_poses > 0 else 0.0,
                    'fused_feature_coverage': poses_with_fused / total_poses if total_poses > 0 else 0.0
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e
    
    def get_fused_features_batch(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve batch of fused features for similarity search or ML training.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of dictionaries with fused features and metadata
        """
        try:
            with self.session_scope() as session:
                query = (
                    select(FusedFeatures, PoseDetection)
                    .join(PoseDetection, FusedFeatures.pose_id == PoseDetection.id)
                    .offset(offset)
                )

                if limit is not None:
                    query = query.limit(limit)

                results = session.execute(query).all()

                return [
                    {
                        'pose_id': pose.id,
                        'person_id': pose.person_id,
                        'confidence': pose.overall_confidence,
                        'fused_vector': fused.fused_vector,
                        'geometric_vector': fused.geometric_vector,
                        'visual_vector': fused.visual_vector,
                        'fusion_method': fused.fusion_method
                    }
                    for fused, pose in results
                ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve fused features: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e

    def get_all_fused_vectors(self) -> np.ndarray:
        """
        Retrieve all fused feature vectors as numpy array.

        Useful for batch similarity search or clustering.

        Returns:
            numpy array of shape (N, FUSED_FEATURE_DIM) where N is number of poses
        """
        try:
            with self.session_scope() as session:
                fused_features = session.execute(
                    select(FusedFeatures.fused_vector)
                ).scalars().all()

                if not fused_features:
                    return np.array([]).reshape(0, settings.FUSED_FEATURE_DIM)

                return np.array([f for f in fused_features], dtype=np.float32)

        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve all fused vectors: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e

    def count_corrected_poses(self) -> int:
        """
        Count the number of manually corrected poses in the database.

        Returns:
            Number of poses with is_corrected=True
        """
        try:
            with self.session_scope() as session:
                count = session.execute(
                    select(func.count(PoseDetection.id)).where(
                        PoseDetection.is_corrected == True
                    )
                ).scalar()

                return count or 0

        except SQLAlchemyError as e:
            logger.error(f"Failed to count corrected poses: {e}", exc_info=True)
            raise StorageManagerError(f"Database error: {e}") from e

    def close(self):
        """
        Close database connections for this instance.

        Note: Shared engine is NOT disposed (other instances may be using it).
        Use close_all_engines() class method to dispose all shared engines.
        """
        # Close all active sessions before removing scoped session
        for session in list(self._active_sessions):
            try:
                session.close()
            except Exception:
                pass  # Suppress errors during cleanup

        # Remove thread-local sessions
        try:
            self.SessionLocal.remove()
        except Exception:
            pass  # Suppress errors during cleanup

        logger.info("StorageManager instance closed (shared engine preserved)")

    @classmethod
    def close_all_engines(cls):
        """
        Dispose all shared database engines.

        WARNING: Only call this when shutting down the entire application.
        All StorageManager instances will lose their database connections.
        """
        with cls._engine_lock:
            for conn_str, engine in cls._engines.items():
                try:
                    engine.dispose()
                    logger.info(f"Disposed shared engine for {conn_str.split('@')[1] if '@' in conn_str else 'unknown'}")
                except Exception as e:
                    logger.error(f"Error disposing engine: {e}")
            cls._engines.clear()
        logger.info("All shared database engines disposed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Suppress errors during cleanup
