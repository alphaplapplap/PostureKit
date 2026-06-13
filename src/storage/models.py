"""
SQLAlchemy ORM models for PostureKit database.
"""
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, ForeignKey, ARRAY, Text, DateTime,
    TIMESTAMP, text, func, CheckConstraint, Index, SmallInteger, LargeBinary
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, BYTEA
from sqlalchemy.orm import declarative_base, relationship, validates
import uuid

from src.config.settings import settings

Base = declarative_base()


def utcnow():
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class Image(Base):
    """Image metadata model."""

    __tablename__ = 'images'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_path = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=True)  # SHA-256 hash for deduplication
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    thumbnail = Column(BYTEA, nullable=True)  # Pre-generated 200x200 JPEG thumbnail
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    pose_detections = relationship(
        'PoseDetection',
        back_populates='image',
        cascade='all, delete-orphan'
    )
    body_parts = relationship(
        'BodyPart',
        back_populates='image',
        cascade='all, delete-orphan'
    )

    __table_args__ = (
        Index('idx_images_created_at', 'created_at'),
        Index('idx_images_file_path', 'file_path', unique=True),
        Index('idx_images_content_hash', 'content_hash'),  # Fast duplicate detection
    )

    def __repr__(self):
        return f"<Image(id={self.id}, path={self.file_path})>"


class BodyPart(Base):
    """
    Body part detection from NudeNet (YOLOv8-based semantic region detector).

    Provides region-level bounding boxes that complement COCO-WholeBody keypoints:
    - Keypoints: Precise joint locations (133 points)
    - Body Parts: Semantic regions with spatial extent (18 classes)

    18 classes: FACE_MALE/FEMALE, BELLY/FEET/ARMPITS/BUTTOCKS (EXPOSED/COVERED),
    FEMALE/MALE_BREAST, FEMALE/MALE_GENITALIA, ANUS (EXPOSED/COVERED)
    """

    __tablename__ = 'body_parts'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.id', ondelete='CASCADE'), nullable=False)
    person_index = Column(Integer, nullable=False, default=0)  # Multi-person support

    # Detection results from NudeNet
    part_name = Column(Text, nullable=False)  # e.g., "FACE_MALE", "FEET_EXPOSED"
    canonical_region = Column(Text, nullable=False)  # Simplified: face, torso, feet, armpits, etc.
    confidence = Column(Float, nullable=False)
    bbox = Column(ARRAY(Float, dimensions=1), nullable=False)  # [x1, y1, x2, y2]
    is_exposed = Column(Boolean, nullable=False, default=False)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    image = relationship('Image', back_populates='body_parts')

    __table_args__ = (
        Index('idx_body_parts_image_id', 'image_id'),
        Index('idx_body_parts_canonical_region', 'canonical_region'),
        Index('idx_body_parts_part_name', 'part_name'),
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='check_confidence_range'),
    )

    def __repr__(self):
        return f"<BodyPart(id={self.id}, part={self.part_name}, conf={self.confidence:.2f})>"


class PoseDetection(Base):
    """Pose detection model with 133 keypoints."""
    
    __tablename__ = 'pose_detections'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey('images.id', ondelete='CASCADE'), nullable=False)
    keypoints = Column(ARRAY(Float), nullable=False)  # 399 elements (133 * 3)
    keypoint_visibility = Column(ARRAY(Integer, dimensions=1), nullable=True)  # 133 elements (0/1/2)
    bbox = Column(ARRAY(Float, dimensions=1), nullable=True)  # [x, y, w, h]
    overall_confidence = Column(Float, nullable=False)
    person_id = Column(Integer, nullable=False)  # Index of person in image

    # Correction tracking
    is_corrected = Column(Boolean, default=False)
    original_keypoints = Column(ARRAY(Float), nullable=True)  # Frozen on first correction
    original_keypoint_visibility = Column(ARRAY(Integer, dimensions=1), nullable=True)
    first_corrected_at = Column(TIMESTAMP(timezone=True), nullable=True)
    correction_count = Column(Integer, default=0)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    image = relationship('Image', back_populates='pose_detections')
    geometric_features = relationship(
        'GeometricFeatures',
        back_populates='pose_detection',
        uselist=False,
        cascade='all, delete-orphan'
    )
    training_label = relationship(
        'TrainingLabel',
        back_populates='pose_detection',
        uselist=False,
        cascade='all, delete-orphan'
    )
    visual_features = relationship(
        'VisualFeatures',
        back_populates='pose_detection',
        uselist=False,
        cascade='all, delete-orphan'
    )
    fused_features = relationship(
        'FusedFeatures',
        back_populates='pose_detection',
        uselist=False,
        cascade='all, delete-orphan'
    )
    correction_statistics = relationship(
        'CorrectionStatistics',
        back_populates='pose_detection',
        uselist=False,
        cascade='all, delete-orphan'
    )
    correction_events = relationship(
        'CorrectionEvent',
        back_populates='pose_detection',
        cascade='all, delete-orphan'
    )

    # Constraints and Indexes
    __table_args__ = (
        CheckConstraint(
            'array_length(keypoints, 1) = 399',
            name='check_keypoints_length'
        ),
        CheckConstraint(
            'keypoint_visibility IS NULL OR array_length(keypoint_visibility, 1) = 133',
            name='check_visibility_length'
        ),
        Index('idx_pose_detections_image_id', 'image_id'),
        Index('idx_pose_detections_corrected', 'is_corrected', postgresql_where=text('is_corrected = true')),
        Index('idx_pose_detections_created_at', 'created_at'),
    )

    def get_keypoints_array(self):
        """
        Get keypoints as (133, 3) numpy array with caching and validation.

        Returns:
            Validated numpy array of shape (133, 3)

        Raises:
            ValueError: If keypoints array is corrupted or malformed
        """
        import numpy as np
        # Check cache
        if not hasattr(self, '_keypoints_cache') or self._keypoints_cache is None:
            # Validate input before reshape to prevent buffer overflow
            if self.keypoints is None:
                raise ValueError(f"Pose {self.id}: keypoints is None")

            kp_array = np.array(self.keypoints, dtype=np.float32)

            # Validate length before reshape (prevents buffer overflow)
            if kp_array.size != 399:
                raise ValueError(
                    f"Pose {self.id}: Expected 399 keypoint elements, got {kp_array.size}. "
                    f"Cannot reshape to (133, 3)"
                )

            # Safe reshape with explicit validation
            try:
                self._keypoints_cache = kp_array.reshape(133, 3)
            except ValueError as e:
                raise ValueError(
                    f"Pose {self.id}: Failed to reshape keypoints from {kp_array.shape} to (133, 3): {e}"
                ) from e

        return self._keypoints_cache

    def get_visibility_array(self):
        """Get visibility flags as (133,) numpy array."""
        import numpy as np
        if self.keypoint_visibility is None:
            # Default: all visible if confidence > 0.3
            kp = self.get_keypoints_array()
            return np.where(kp[:, 2] > 0.3, 2, 1).astype(np.int32)
        return np.array(self.keypoint_visibility, dtype=np.int32)

    def set_keypoints(self, keypoints):
        """Set keypoints from (133, 3) numpy array."""
        import numpy as np
        assert keypoints.shape == (133, 3), f"Expected (133, 3), got {keypoints.shape}"
        self.keypoints = keypoints.ravel().tolist()  # ravel is faster than flatten
        # Invalidate cache
        self._keypoints_cache = None

    def set_visibility(self, visibility):
        """Set visibility from (133,) numpy array."""
        import numpy as np
        assert visibility.shape == (133,), f"Expected (133,), got {visibility.shape}"
        assert np.all(np.isin(visibility, [0, 1, 2])), "Visibility must be 0, 1, or 2"
        self.keypoint_visibility = visibility.tolist()

    def __repr__(self):
        return f"<PoseDetection(id={self.id}, person={self.person_id}, conf={self.overall_confidence:.3f})>"


class GeometricFeatures(Base):
    """Geometric features model for pose similarity."""

    __tablename__ = 'geometric_features'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pose_id = Column(UUID(as_uuid=True), ForeignKey('pose_detections.id', ondelete='CASCADE'), nullable=False, unique=True)
    feature_vector = Column(ARRAY(Float, dimensions=1), nullable=False)  # 52 dimensions
    feature_confidence = Column(ARRAY(Float, dimensions=1), nullable=True)  # 52 confidence scores (0-1)
    joint_angles = Column(JSONB, nullable=True)
    limb_ratios = Column(JSONB, nullable=True)
    body_angles = Column(JSONB, nullable=True)
    symmetry_scores = Column(JSONB, nullable=True)
    occlusion_pattern = Column(ARRAY(Float, dimensions=1), nullable=True)  # 7 dimensions
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    pose_detection = relationship('PoseDetection', back_populates='geometric_features')
    fused_features = relationship(
        'FusedFeatures',
        back_populates='geometric_features',
        uselist=False,
        cascade='all, delete-orphan'
    )

    # pose_id is unique=True, which is backed by a unique index
    # (geometric_features_pose_id_key) that serves all pose_id lookups. A
    # separate idx_geometric_features_pose_id would be a strict duplicate, so it
    # is intentionally NOT declared here (and dropped from existing DBs via
    # migrations/002_drop_duplicate_pose_id_indexes.sql).

    def __repr__(self):
        return f"<GeometricFeatures(id={self.id}, pose_id={self.pose_id})>"


class VisualFeatures(Base):
    """Visual feature vectors from MobileNetV3."""

    __tablename__ = 'visual_features'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pose_id = Column(UUID(as_uuid=True), ForeignKey('pose_detections.id', ondelete='CASCADE'), nullable=False, unique=True)
    feature_vector = Column(ARRAY(Float, dimensions=1), nullable=False)  # 576-dim
    model_name = Column(String, nullable=False, default='mobilenet_v3_small')
    normalization = Column(String, nullable=True, default='l2')
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    pose_detection = relationship('PoseDetection', back_populates='visual_features')
    fused_features = relationship(
        'FusedFeatures',
        back_populates='visual_features',
        uselist=False,
        cascade='all, delete-orphan'
    )

    # pose_id unique=True is backed by visual_features_pose_id_key; the former
    # idx_visual_features_pose_id was a duplicate (dropped via migration 002).

    @validates('feature_vector')
    def validate_feature_vector(self, key, value):
        """Validate feature vector has exactly VISUAL_FEATURE_DIM dimensions."""
        expected_dim = settings.VISUAL_FEATURE_DIM
        if value is not None and len(value) != expected_dim:
            raise ValueError(f"Feature vector must have {expected_dim} dimensions, got {len(value)}")
        return value

    def __repr__(self):
        return f"<VisualFeatures(id={self.id}, pose_id={self.pose_id}, model={self.model_name})>"


class FusedFeatures(Base):
    """Fused multimodal features combining geometric and visual."""

    __tablename__ = 'fused_features'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pose_id = Column(UUID(as_uuid=True), ForeignKey('pose_detections.id', ondelete='CASCADE'), nullable=False, unique=True)
    geometric_feature_id = Column(UUID(as_uuid=True), ForeignKey('geometric_features.id', ondelete='CASCADE'), nullable=False)
    visual_feature_id = Column(UUID(as_uuid=True), ForeignKey('visual_features.id', ondelete='CASCADE'), nullable=False)
    fused_vector = Column(ARRAY(Float, dimensions=1), nullable=False)  # 628-dim (52 + 576)
    geometric_vector = Column(ARRAY(Float, dimensions=1), nullable=False)  # 52-dim
    visual_vector = Column(ARRAY(Float, dimensions=1), nullable=False)  # 576-dim
    fusion_method = Column(String, nullable=False, default='concatenate')
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    pose_detection = relationship('PoseDetection', back_populates='fused_features')
    geometric_features = relationship('GeometricFeatures', back_populates='fused_features')
    visual_features = relationship('VisualFeatures', back_populates='fused_features')

    __table_args__ = (
        CheckConstraint(
            "fusion_method IN ('concatenate', 'weighted', 'normalized')",
            name='fusion_method_valid'
        ),
        # pose_id unique=True is backed by fused_features_pose_id_key; the former
        # idx_fused_features_pose_id was a duplicate (dropped via migration 002).
        # The geometric/visual FK indexes are NOT unique, so they are kept.
        Index('idx_fused_features_geometric_id', 'geometric_feature_id'),
        Index('idx_fused_features_visual_id', 'visual_feature_id'),
    )

    @validates('fused_vector')
    def validate_fused_vector(self, key, value):
        """Validate fused vector has exactly FUSED_FEATURE_DIM dimensions."""
        expected_dim = settings.FUSED_FEATURE_DIM
        if value is not None and len(value) != expected_dim:
            raise ValueError(f"Fused vector must have {expected_dim} dimensions, got {len(value)}")
        return value

    @validates('geometric_vector')
    def validate_geometric_vector(self, key, value):
        """Validate geometric vector has exactly GEOMETRIC_FEATURE_DIM dimensions."""
        expected_dim = settings.GEOMETRIC_FEATURE_DIM
        if value is not None and len(value) != expected_dim:
            raise ValueError(f"Geometric vector must have {expected_dim} dimensions, got {len(value)}")
        return value

    @validates('visual_vector')
    def validate_visual_vector(self, key, value):
        """Validate visual vector has exactly VISUAL_FEATURE_DIM dimensions."""
        expected_dim = settings.VISUAL_FEATURE_DIM
        if value is not None and len(value) != expected_dim:
            raise ValueError(f"Visual vector must have {expected_dim} dimensions, got {len(value)}")
        return value

    @validates('fusion_method')
    def validate_fusion_method(self, key, value):
        """Validate fusion method is one of the allowed values."""
        allowed_methods = ['concatenate', 'weighted', 'normalized']
        if value not in allowed_methods:
            raise ValueError(f"Fusion method must be one of {allowed_methods}, got {value}")
        return value

    def __repr__(self):
        return f"<FusedFeatures(id={self.id}, pose_id={self.pose_id}, method={self.fusion_method})>"


class TrainingLabel(Base):
    """Training labels and annotations model."""

    __tablename__ = 'training_labels'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pose_id = Column(UUID(as_uuid=True), ForeignKey('pose_detections.id', ondelete='CASCADE'), nullable=False)
    category = Column(Text, nullable=True)
    difficulty = Column(Text, nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    user_notes = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    pose_detection = relationship('PoseDetection', back_populates='training_label')

    __table_args__ = (
        Index('idx_training_labels_pose_id', 'pose_id'),
        Index('idx_training_labels_category', 'category'),
    )

    def __repr__(self):
        return f"<TrainingLabel(id={self.id}, category={self.category})>"


class CorrectionStatistics(Base):
    """
    Statistics about manual corrections made to a pose.

    Attributes:
        id: Primary key
        pose_id: Foreign key to pose_detections
        keypoints_corrected_count: Number of keypoints that were moved
        avg_correction_distance: Average distance keypoints moved (pixels)
        max_correction_distance: Maximum distance any keypoint moved
        corrected_keypoint_types: List of keypoint names that were corrected
        avg_confidence_before: Average keypoint confidence before correction
        avg_confidence_after: Average keypoint confidence after correction
        created_at: Timestamp
    """
    __tablename__ = 'correction_statistics'

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    pose_id = Column(UUID(as_uuid=True), ForeignKey('pose_detections.id', ondelete='CASCADE'), unique=True, nullable=False)
    keypoints_corrected_count = Column(Integer, nullable=False)
    avg_correction_distance = Column(Float)
    max_correction_distance = Column(Float)
    corrected_keypoint_types = Column(ARRAY(String))
    avg_confidence_before = Column(Float)
    avg_confidence_after = Column(Float)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationship
    pose_detection = relationship('PoseDetection', back_populates='correction_statistics')

    __table_args__ = (
        # pose_id unique=True is backed by correction_statistics_pose_id_key; the
        # former idx_correction_statistics_pose_id was a duplicate (dropped via
        # migration 002). The GIN index on corrected_keypoint_types is kept.
        Index('idx_correction_statistics_keypoint_types', 'corrected_keypoint_types', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<CorrectionStatistics(id={self.id}, pose_id={self.pose_id}, corrected={self.keypoints_corrected_count})>"


class CorrectionEvent(Base):
    """
    Log of correction events (append-only history).

    Attributes:
        id: Primary key
        pose_id: Foreign key to pose_detections
        correction_number: Sequential correction number for this pose
        changed_keypoint_indices: Array of keypoint indices that were moved
        max_displacement: Largest movement in pixels
        avg_displacement: Average movement across changed keypoints
        timestamp: When this correction occurred
    """
    __tablename__ = 'correction_events'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pose_id = Column(UUID(as_uuid=True), ForeignKey('pose_detections.id', ondelete='CASCADE'), nullable=False)
    correction_number = Column(Integer, nullable=False)
    changed_keypoint_indices = Column(ARRAY(SmallInteger))  # Match schema: SMALLINT[]
    max_displacement = Column(Float)
    avg_displacement = Column(Float)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationship
    pose_detection = relationship('PoseDetection', back_populates='correction_events')

    __table_args__ = (
        Index('idx_correction_events_pose_id', 'pose_id'),
    )

    def __repr__(self):
        return f"<CorrectionEvent(id={self.id}, pose_id={self.pose_id}, correction_number={self.correction_number})>"


class ExcludedFolder(Base):
    """
    Folder excluded from indexing and search results.

    Files under an excluded folder are skipped during indexing, and
    already-indexed images inside one are filtered out of search/browse
    results (no re-index required).

    Attributes:
        id: Primary key
        folder_path: Absolute path to the excluded folder (unique)
        created_at: When the exclusion was added
        notes: Optional user notes about why the folder is excluded
    """
    __tablename__ = 'excluded_folders'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    folder_path = Column(Text, nullable=False, unique=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    notes = Column(Text)

    __table_args__ = (
        Index('idx_excluded_folders_path', 'folder_path'),
    )

    def __repr__(self):
        return f"<ExcludedFolder(id={self.id}, folder_path={self.folder_path})>"
