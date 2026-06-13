"""
Core data models for PostureKit
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import numpy as np
import json


@dataclass
class Keypoint:
    """Represents a single keypoint in the pose"""
    index: int
    name: str
    x: float
    y: float
    confidence: float
    visibility: int  # 0=missing, 1=occluded, 2=visible
    is_corrected: bool = False
    original_x: Optional[float] = None
    original_y: Optional[float] = None

    def __post_init__(self):
        """Store original position on creation"""
        if self.original_x is None:
            self.original_x = self.x
        if self.original_y is None:
            self.original_y = self.y

    def reset(self):
        """Reset to original position"""
        if self.original_x is not None:
            self.x = self.original_x
        if self.original_y is not None:
            self.y = self.original_y
        self.is_corrected = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'index': self.index,
            'name': self.name,
            'x': self.x,
            'y': self.y,
            'confidence': self.confidence,
            'visibility': self.visibility,
            'is_corrected': self.is_corrected,
            'original_x': self.original_x,
            'original_y': self.original_y
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Keypoint':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class Person:
    """Represents a detected person with pose"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    keypoints: List[Keypoint]
    confidence: float
    total_count: int = 1  # Total persons in image
    labels: Dict[str, Any] = field(default_factory=dict)

    @property
    def visible_keypoints(self) -> int:
        """Count of visible keypoints"""
        return sum(1 for kp in self.keypoints if kp.visibility > 0)

    @property
    def occlusion_percentage(self) -> float:
        """Calculate occlusion percentage"""
        occluded = sum(1 for kp in self.keypoints if kp.visibility < 2)
        return (occluded / len(self.keypoints)) * 100 if self.keypoints else 0

    @property
    def is_modified(self) -> bool:
        """Check if any keypoint has been corrected"""
        return any(kp.is_corrected for kp in self.keypoints)

    def resetToOriginal(self):
        """Reset all keypoints to original positions"""
        for kp in self.keypoints:
            kp.reset()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'keypoints': [kp.to_dict() for kp in self.keypoints],
            'confidence': self.confidence,
            'total_count': self.total_count,
            'labels': self.labels
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Person':
        """Create from dictionary"""
        data['keypoints'] = [Keypoint.from_dict(kp) for kp in data['keypoints']]
        return cls(**data)


@dataclass
class Pose:
    """Complete pose detection result for an image"""
    id: Optional[int] = None
    image_path: str = ""
    image_width: int = 0
    image_height: int = 0
    persons: List[Person] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "rtmw-l_cocktail14"
    is_modified: bool = False
    geometric_features: Optional[np.ndarray] = None
    viewpoint: Optional[Dict[str, float]] = None

    @property
    def person_count(self) -> int:
        """Number of detected persons"""
        return len(self.persons)

    def resetToOriginal(self):
        """Reset all persons to original detections"""
        for person in self.persons:
            person.resetToOriginal()
        self.is_modified = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'image_path': self.image_path,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'persons': [p.to_dict() for p in self.persons],
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version,
            'is_modified': self.is_modified,
            'viewpoint': self.viewpoint
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pose':
        """Create from dictionary"""
        data['persons'] = [Person.from_dict(p) for p in data['persons']]
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def to_coco_format(self) -> Dict[str, Any]:
        """Convert to COCO format for export"""
        annotations = []
        for person in self.persons:
            keypoints_flat = []
            for kp in person.keypoints:
                keypoints_flat.extend([kp.x, kp.y, kp.visibility])

            annotations.append({
                'id': person.id,
                'image_id': self.id,
                'category_id': 1,  # Person category
                'bbox': list(person.bbox),
                'keypoints': keypoints_flat,
                'num_keypoints': person.visible_keypoints,
                'score': person.confidence,
                'area': person.bbox[2] * person.bbox[3]
            })

        return {
            'image': {
                'id': self.id,
                'file_name': self.image_path,
                'width': self.image_width,
                'height': self.image_height
            },
            'annotations': annotations
        }


# Keypoint names for COCO-WholeBody format (133 keypoints)
KEYPOINT_NAMES = {
    # Body (17 keypoints)
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",

    # Feet (6 keypoints)
    17: "Left Big Toe",
    18: "Left Small Toe",
    19: "Left Heel",
    20: "Right Big Toe",
    21: "Right Small Toe",
    22: "Right Heel",

    # Face (68 keypoints) - COCO-WholeBody format
    # Jawline (17 points)
    23: "Face 1", 24: "Face 2", 25: "Face 3", 26: "Face 4", 27: "Face 5",
    28: "Face 6", 29: "Face 7", 30: "Face 8", 31: "Face 9", 32: "Face 10",
    33: "Face 11", 34: "Face 12", 35: "Face 13", 36: "Face 14", 37: "Face 15",
    38: "Face 16", 39: "Face 17",
    # Left eyebrow (5 points)
    40: "Left Eyebrow 1", 41: "Left Eyebrow 2", 42: "Left Eyebrow 3",
    43: "Left Eyebrow 4", 44: "Left Eyebrow 5",
    # Right eyebrow (5 points)
    45: "Right Eyebrow 1", 46: "Right Eyebrow 2", 47: "Right Eyebrow 3",
    48: "Right Eyebrow 4", 49: "Right Eyebrow 5",
    # Nose (9 points)
    50: "Nose Bridge 1", 51: "Nose Bridge 2", 52: "Nose Bridge 3", 53: "Nose Bridge 4",
    54: "Nose Tip", 55: "Left Nostril", 56: "Right Nostril",
    57: "Nose Bottom 1", 58: "Nose Bottom 2",
    # Left eye (6 points)
    59: "Left Eye 1", 60: "Left Eye 2", 61: "Left Eye 3",
    62: "Left Eye 4", 63: "Left Eye 5", 64: "Left Eye 6",
    # Right eye (6 points)
    65: "Right Eye 1", 66: "Right Eye 2", 67: "Right Eye 3",
    68: "Right Eye 4", 69: "Right Eye 5", 70: "Right Eye 6",
    # Outer mouth (12 points)
    71: "Mouth Corner Left", 72: "Upper Lip 1", 73: "Upper Lip 2", 74: "Upper Lip 3",
    75: "Mouth Corner Right", 76: "Lower Lip 1", 77: "Lower Lip 2", 78: "Lower Lip 3",
    79: "Mouth 9", 80: "Mouth 10", 81: "Mouth 11", 82: "Mouth 12",
    # Inner mouth (8 points)
    83: "Inner Mouth 1", 84: "Inner Mouth 2", 85: "Inner Mouth 3", 86: "Inner Mouth 4",
    87: "Inner Mouth 5", 88: "Inner Mouth 6", 89: "Inner Mouth 7", 90: "Inner Mouth 8",

    # Left hand (21 keypoints) - indices 91-111
    91: "Left Wrist Base",
    # Thumb
    92: "Left Thumb 1 (CMC)", 93: "Left Thumb 2 (MCP)", 94: "Left Thumb 3 (IP)", 95: "Left Thumb Tip",
    # Index finger
    96: "Left Index 1 (MCP)", 97: "Left Index 2 (PIP)", 98: "Left Index 3 (DIP)", 99: "Left Index Tip",
    # Middle finger
    100: "Left Middle 1 (MCP)", 101: "Left Middle 2 (PIP)", 102: "Left Middle 3 (DIP)", 103: "Left Middle Tip",
    # Ring finger
    104: "Left Ring 1 (MCP)", 105: "Left Ring 2 (PIP)", 106: "Left Ring 3 (DIP)", 107: "Left Ring Tip",
    # Pinky finger
    108: "Left Pinky 1 (MCP)", 109: "Left Pinky 2 (PIP)", 110: "Left Pinky 3 (DIP)", 111: "Left Pinky Tip",

    # Right hand (21 keypoints) - indices 112-132
    112: "Right Wrist Base",
    # Thumb
    113: "Right Thumb 1 (CMC)", 114: "Right Thumb 2 (MCP)", 115: "Right Thumb 3 (IP)", 116: "Right Thumb Tip",
    # Index finger
    117: "Right Index 1 (MCP)", 118: "Right Index 2 (PIP)", 119: "Right Index 3 (DIP)", 120: "Right Index Tip",
    # Middle finger
    121: "Right Middle 1 (MCP)", 122: "Right Middle 2 (PIP)", 123: "Right Middle 3 (DIP)", 124: "Right Middle Tip",
    # Ring finger
    125: "Right Ring 1 (MCP)", 126: "Right Ring 2 (PIP)", 127: "Right Ring 3 (DIP)", 128: "Right Ring Tip",
    # Pinky finger
    129: "Right Pinky 1 (MCP)", 130: "Right Pinky 2 (PIP)", 131: "Right Pinky 3 (DIP)", 132: "Right Pinky Tip",
}

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Feet
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22)
]


@dataclass
class PoseResult:
    """
    Result from pose detection on a single person.

    Defined HERE (a torch-free module) rather than in pose_detector.py so that
    consumers needing only the dataclass — e.g. the geometric feature extractor and
    the search engine's flip-search re-extraction — can import it WITHOUT dragging in
    torch/mmpose. The search server (skip_models, faiss-only) must never load torch:
    a second OpenMP runtime (torch's libomp) alongside faiss's aborts the process
    (OMP Error #15). pose_detector re-exports this for backward compatibility.

    Attributes:
        keypoints: Array of shape (133, 3) where each row is [x, y, confidence]
        visibility: Array of shape (133,) with COCO visibility flags:
                   0 = not labeled (not visible in image)
                   1 = labeled but occluded (person present but keypoint hidden)
                   2 = labeled and visible (keypoint clearly visible)
        bbox: Bounding box [x, y, width, height]
        overall_confidence: Mean confidence across all keypoints
        person_id: Index of person in image (0-based)
    """
    keypoints: np.ndarray  # (133, 3)
    visibility: np.ndarray  # (133,)
    bbox: np.ndarray       # (4,) [x, y, w, h]
    overall_confidence: float
    person_id: int

    def __post_init__(self):
        """Validate data after initialization."""
        assert self.keypoints.shape == (133, 3), \
            f"Expected keypoints shape (133, 3), got {self.keypoints.shape}"
        assert self.visibility.shape == (133,), \
            f"Expected visibility shape (133,), got {self.visibility.shape}"
        # Allow continuous visibility values [0, 2] for ensemble fusion
        assert np.all((self.visibility >= 0) & (self.visibility <= 2)), \
            f"Visibility must be in [0, 2], got range [{np.min(self.visibility)}, {np.max(self.visibility)}]"
        assert self.bbox.shape == (4,), \
            f"Expected bbox shape (4,), got {self.bbox.shape}"
        assert 0.0 <= self.overall_confidence <= 1.0, \
            f"Confidence must be in [0, 1], got {self.overall_confidence}"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'keypoints': self.keypoints.tolist(),
            'visibility': self.visibility.tolist(),
            'bbox': self.bbox.tolist(),
            'overall_confidence': float(self.overall_confidence),
            'person_id': self.person_id,
        }

    def get_visible_keypoints(self) -> np.ndarray:
        """Get only visible keypoints (visibility == 2)."""
        return self.keypoints[self.visibility == 2]

    def count_visible(self) -> int:
        """Count visible keypoints."""
        return int((self.visibility == 2).sum())

    def count_occluded(self) -> int:
        """Count occluded keypoints."""
        return int((self.visibility == 1).sum())
