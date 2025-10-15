"""
Similarity search data structures for PostureKit.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from uuid import UUID


@dataclass
class EnrichedSimilarPose:
    """
    Similar pose enriched with database metadata.

    Attributes:
        pose_id: UUID of the similar pose
        distance: L2 distance in feature space
        similarity_score: Normalized similarity (0-1, higher = more similar)
        confidence: Pose detection confidence
        category: Training label category (if labeled)
        tags: Training label tags (if labeled)
        image_path: Path to source image
    """
    pose_id: UUID
    distance: float
    similarity_score: float
    confidence: float = 0.0
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    image_path: Optional[str] = None
