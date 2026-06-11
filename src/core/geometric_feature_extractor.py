"""
Feature Extractor for PostureKit.
Computes 52-dimensional geometric feature vectors from pose keypoints.
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

from src.core.pose_detector import PoseResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GeometricFeatures:
    """
    Computed geometric features from a pose.

    Attributes:
        feature_vector: 52-dimensional feature array
        feature_confidence: 52-dimensional confidence array (0-1 per feature)
        joint_angles: Dictionary of computed joint angles
        limb_ratios: Dictionary of limb length ratios
        body_angles: Dictionary of body part orientations
        symmetry_scores: Dictionary of left-right symmetry measures
        occlusion_pattern: Binary array indicating visible keypoints
    """
    feature_vector: np.ndarray  # (52,)
    feature_confidence: np.ndarray  # (52,) confidence scores 0-1
    joint_angles: Dict[str, float]
    limb_ratios: Dict[str, float]
    body_angles: Dict[str, float]
    symmetry_scores: Dict[str, float]
    occlusion_pattern: np.ndarray  # (7,)

    def __post_init__(self):
        """Validate feature dimensions."""
        assert self.feature_vector.shape == (52,), \
            f"Feature vector must be (52,), got {self.feature_vector.shape}"
        assert self.feature_vector.dtype == np.float32, \
            f"Feature vector must be float32, got {self.feature_vector.dtype}"
        assert self.feature_confidence.shape == (52,), \
            f"Feature confidence must be (52,), got {self.feature_confidence.shape}"
        assert self.feature_confidence.dtype == np.float32, \
            f"Feature confidence must be float32, got {self.feature_confidence.dtype}"
        assert self.occlusion_pattern.shape == (7,), \
            f"Occlusion pattern must be (7,), got {self.occlusion_pattern.shape}"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'feature_vector': self.feature_vector.tolist(),
            'feature_confidence': self.feature_confidence.tolist(),
            'joint_angles': self.joint_angles,
            'limb_ratios': self.limb_ratios,
            'body_angles': self.body_angles,
            'symmetry_scores': self.symmetry_scores,
            'occlusion_pattern': self.occlusion_pattern.tolist()
        }


class FeatureExtractorError(Exception):
    """Base exception for Feature Extractor errors."""
    pass


class GeometricFeatureExtractor:
    """
    Extracts 52-dimensional geometric feature vectors from poses.

    Feature breakdown (52 dimensions total):
    - Joint angles (12): elbow, knee, shoulder, hip angles
    - Limb length ratios (10): normalized by body height
    - Body part angles (15): torso twist, limb orientations
    - Symmetry scores (8): left vs right body parts
    - Occlusion pattern (7): encoded visibility of major body regions

    Attributes:
        confidence_threshold: Minimum confidence for valid keypoint
        normalize: Whether to normalize features to [0, 1] range
    """

    # Weight of the binary occlusion-pattern dims (45-51) in the normalized
    # vector. At 1.0 a single flipped flag contributes the same squared L2 as a
    # 180-degree joint-angle error and dominates the metric. 0.25 was meant to
    # be "informative, not dominant", but on occluded pairs — where half the
    # positional dims are masked while flags never are — the 7 flags still
    # accounted for ~23% of pair divergence (measured on 94 real near-neighbor
    # occluded pairs, sim_occlusion_params.py 2026-06-12). 0.15 (~27-degree
    # equivalent) keeps visibility informative without outvoting positioning.
    OCCLUSION_FLAG_WEIGHT = 0.15

    # Exponent of the visibility-trust curve in _get_keypoint_confidence.
    # 2.0 (the 2026-06-10 occlusion-repair keystone) crushed vis=1 keypoints
    # below every masked-search gate — which also erased LEGS, the most
    # frequently occluded body part, from comparison: occluded pairs matched on
    # a median of 4/17 leg dims. Linear (1.0) admits the model's high-confidence
    # occluded estimates (conf*0.5, so >= 0.70 raw passes a 0.35 gate) and
    # doubles leg coverage to 8/17 while still halving the trust of estimated
    # positions. Validated against the synthetic-occlusion self-similarity
    # benchmark before adoption.
    VIS_TRUST_EXPONENT = 1.0

    # Keypoint indices (RTMW-L 133-keypoint model)
    KEYPOINT_NOSE = 0
    KEYPOINT_LEFT_EYE = 1
    KEYPOINT_RIGHT_EYE = 2
    KEYPOINT_LEFT_EAR = 3
    KEYPOINT_RIGHT_EAR = 4
    KEYPOINT_LEFT_SHOULDER = 5
    KEYPOINT_RIGHT_SHOULDER = 6
    KEYPOINT_LEFT_ELBOW = 7
    KEYPOINT_RIGHT_ELBOW = 8
    KEYPOINT_LEFT_WRIST = 9
    KEYPOINT_RIGHT_WRIST = 10
    KEYPOINT_LEFT_HIP = 11
    KEYPOINT_RIGHT_HIP = 12
    KEYPOINT_LEFT_KNEE = 13
    KEYPOINT_RIGHT_KNEE = 14
    KEYPOINT_LEFT_ANKLE = 15
    KEYPOINT_RIGHT_ANKLE = 16

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        normalize: bool = True,
        use_occluded_keypoints: bool = True
    ):
        """
        Initialize Feature Extractor.

        Args:
            confidence_threshold: Minimum confidence for valid keypoint
            normalize: Whether to normalize features to [0, 1]
            use_occluded_keypoints: Whether to use keypoints with visibility=1 (occluded but present).
                                   True (default): Uses occluded keypoints if confidence >= threshold.
                                                  Rationale: Model's inferred positions still informative.
                                   False: Only uses fully visible keypoints (visibility=2).
                                         Rationale: Conservative, avoids potentially unreliable positions.
                                   See docs/DESIGN_DECISION_visibility_filtering.md for detailed analysis.
        """
        self.confidence_threshold = confidence_threshold
        self.normalize = normalize
        self.use_occluded_keypoints = use_occluded_keypoints
        self._current_visibility = None  # Set during extract()

        logger.info(
            f"GeometricFeatureExtractor initialized",
            extra={'extra_data': {
                'confidence_threshold': confidence_threshold,
                'normalize': normalize,
                'use_occluded_keypoints': use_occluded_keypoints
            }}
        )

    def _get_keypoint(
        self,
        keypoints: np.ndarray,
        index: int
    ) -> Optional[np.ndarray]:
        """
        Get keypoint if valid according to visibility and confidence policies.

        Returns keypoint [x, y] if:
        - use_occluded_keypoints=True: visibility >= 1.0 AND confidence >= threshold
        - use_occluded_keypoints=False: visibility >= 1.5 AND confidence >= threshold

        Note:
            Handles continuous visibility (0.0-2.0) from ensemble fusion.

        Args:
            keypoints: (133, 3) array of [x, y, confidence]
            index: Keypoint index to retrieve

        Returns:
            [x, y] coordinates if valid, None otherwise
        """
        if self._current_visibility is None:
            # Fallback: no visibility data, use confidence only (backward compatible)
            if keypoints[index, 2] >= self.confidence_threshold:
                return keypoints[index, :2]
            return None

        vis = float(self._current_visibility[index])
        conf = keypoints[index, 2]

        if self.use_occluded_keypoints:
            # Permissive: use visible OR occluded keypoints if confident
            if vis >= 1.0 and conf >= self.confidence_threshold:
                return keypoints[index, :2]
        else:
            # Strict: only use mostly visible keypoints (excludes purely occluded)
            if vis >= 1.5 and conf >= self.confidence_threshold:
                return keypoints[index, :2]

        return None

    def _calculate_angle(
        self,
        p1: Optional[np.ndarray],
        p2: Optional[np.ndarray],
        p3: Optional[np.ndarray]
    ) -> Optional[float]:
        """
        Calculate angle at p2 formed by p1-p2-p3.

        Args:
            p1, p2, p3: Points as [x, y] arrays

        Returns:
            Angle in degrees (0-180), or None if any point is missing
        """
        if p1 is None or p2 is None or p3 is None:
            return None

        # Vectors from p2 to p1 and p3
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi

        return float(angle)

    def _calculate_distance(
        self,
        p1: Optional[np.ndarray],
        p2: Optional[np.ndarray]
    ) -> Optional[float]:
        """Calculate Euclidean distance between two points."""
        if p1 is None or p2 is None:
            return None
        return float(np.linalg.norm(p1 - p2))

    def _get_keypoint_confidence(
        self,
        keypoints: np.ndarray,
        indices: List[int]
    ) -> float:
        """
        Get minimum confidence from a list of keypoint indices.

        Args:
            keypoints: (133, 3) array of [x, y, confidence]
            indices: List of keypoint indices to check

        Returns:
            Minimum confidence among keypoints, or 0.0 if any is invalid

        Note:
            Handles both discrete visibility (0/1/2) and continuous visibility (0.0-2.0)
            from ensemble fusion:
            - 0.0 = missing
            - 1.0 = occluded
            - 2.0 = visible
            - 0.5-2.0 = interpolated states (ensemble fusion)
        """
        confidences = []
        for idx in indices:
            # Check if keypoint is valid (using same logic as _get_keypoint)
            if self._current_visibility is not None:
                vis = float(self._current_visibility[idx])
                conf = keypoints[idx, 2]

                # Scale trust by visibility: an occluded-but-inferred keypoint
                # (vis=1) keeps its geometry in the feature VALUE but loses
                # confidence — the model's raw score stays high under occlusion
                # (~0.9), so visibility is the more sensitive signal. The curve
                # exponent is the policy knob (see VIS_TRUST_EXPONENT): linear
                # gives vis=1 trust 0.5, so only the model's most confident
                # occluded estimates (raw >= 0.70) cross the default 0.35
                # masked-search gate; vis=2 keypoints are untouched either way.
                vis_trust = (min(vis, 2.0) / 2.0) ** self.VIS_TRUST_EXPONENT

                if self.use_occluded_keypoints:
                    # Accept occluded or better (vis >= 1.0)
                    if vis >= 1.0 and conf >= self.confidence_threshold:
                        confidences.append(conf * vis_trust)
                    else:
                        return 0.0  # Invalid keypoint, feature is unreliable
                else:
                    # Accept only mostly visible keypoints (vis >= 1.5)
                    # This excludes purely occluded (vis=1.0) but includes partially visible
                    if vis >= 1.5 and conf >= self.confidence_threshold:
                        confidences.append(conf * vis_trust)
                    else:
                        return 0.0
            else:
                # Fallback: no visibility data
                conf = keypoints[idx, 2]
                if conf >= self.confidence_threshold:
                    confidences.append(conf)
                else:
                    return 0.0

        return float(np.min(confidences)) if confidences else 0.0

    def _extract_joint_angles(self, keypoints: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract 12 joint angles with confidence tracking.

        Returns:
            Tuple of (angles_dict, confidence_dict)
            - angles_dict: Joint angle features (default 90° if missing)
            - confidence_dict: Confidence per angle (0.0 if any keypoint invalid)
        """
        # Get keypoints
        l_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_SHOULDER)
        r_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_SHOULDER)
        l_elbow = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ELBOW)
        r_elbow = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ELBOW)
        l_wrist = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_WRIST)
        r_wrist = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_WRIST)
        l_hip = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_HIP)
        r_hip = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_HIP)
        l_knee = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_KNEE)
        r_knee = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_KNEE)
        l_ankle = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ANKLE)
        r_ankle = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ANKLE)

        angles = {
            'left_elbow': self._calculate_angle(l_shoulder, l_elbow, l_wrist) or 90.0,
            'right_elbow': self._calculate_angle(r_shoulder, r_elbow, r_wrist) or 90.0,
            'left_shoulder': self._calculate_angle(l_elbow, l_shoulder, l_hip) or 90.0,
            'right_shoulder': self._calculate_angle(r_elbow, r_shoulder, r_hip) or 90.0,
            'left_hip': self._calculate_angle(l_shoulder, l_hip, l_knee) or 90.0,
            'right_hip': self._calculate_angle(r_shoulder, r_hip, r_knee) or 90.0,
            'left_knee': self._calculate_angle(l_hip, l_knee, l_ankle) or 180.0,
            'right_knee': self._calculate_angle(r_hip, r_knee, r_ankle) or 180.0,
            'left_armpit': self._calculate_angle(l_elbow, l_shoulder, r_shoulder) or 90.0,
            'right_armpit': self._calculate_angle(r_elbow, r_shoulder, l_shoulder) or 90.0,
            'left_leg_spread': self._calculate_angle(l_knee, l_hip, r_hip) or 90.0,
            'right_leg_spread': self._calculate_angle(r_knee, r_hip, l_hip) or 90.0,
        }

        # Track confidence for each angle
        confidences = {
            'left_elbow': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_LEFT_ELBOW, self.KEYPOINT_LEFT_WRIST]),
            'right_elbow': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_RIGHT_ELBOW, self.KEYPOINT_RIGHT_WRIST]),
            'left_shoulder': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_ELBOW, self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_LEFT_HIP]),
            'right_shoulder': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_ELBOW, self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_RIGHT_HIP]),
            'left_hip': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_LEFT_HIP, self.KEYPOINT_LEFT_KNEE]),
            'right_hip': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_RIGHT_HIP, self.KEYPOINT_RIGHT_KNEE]),
            'left_knee': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_HIP, self.KEYPOINT_LEFT_KNEE, self.KEYPOINT_LEFT_ANKLE]),
            'right_knee': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_HIP, self.KEYPOINT_RIGHT_KNEE, self.KEYPOINT_RIGHT_ANKLE]),
            'left_armpit': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_ELBOW, self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_RIGHT_SHOULDER]),
            'right_armpit': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_ELBOW, self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_LEFT_SHOULDER]),
            'left_leg_spread': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_KNEE, self.KEYPOINT_LEFT_HIP, self.KEYPOINT_RIGHT_HIP]),
            'right_leg_spread': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_KNEE, self.KEYPOINT_RIGHT_HIP, self.KEYPOINT_LEFT_HIP]),
        }

        return angles, confidences

    def _calculate_body_height(self, keypoints: np.ndarray) -> Optional[float]:
        """Calculate approximate body height for normalization."""
        nose = self._get_keypoint(keypoints, self.KEYPOINT_NOSE)
        l_ankle = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ANKLE)
        r_ankle = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ANKLE)

        heights = []
        if nose is not None and l_ankle is not None:
            heights.append(self._calculate_distance(nose, l_ankle))
        if nose is not None and r_ankle is not None:
            heights.append(self._calculate_distance(nose, r_ankle))

        return np.mean(heights) if heights else None

    def _extract_limb_ratios(self, keypoints: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract 10 limb length ratios normalized by body height with confidence tracking.

        Returns:
            Tuple of (ratios_dict, confidence_dict)
            - ratios_dict: Limb ratio features (default 0.25 if missing)
            - confidence_dict: Confidence per ratio (0.0 if any endpoint invalid)
        """
        body_height = self._calculate_body_height(keypoints)
        height_valid = body_height is not None
        body_height = body_height or 100.0

        # Get keypoints
        l_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_SHOULDER)
        r_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_SHOULDER)
        l_elbow = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ELBOW)
        r_elbow = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ELBOW)
        l_wrist = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_WRIST)
        r_wrist = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_WRIST)
        l_hip = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_HIP)
        r_hip = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_HIP)
        l_knee = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_KNEE)
        r_knee = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_KNEE)
        l_ankle = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ANKLE)
        r_ankle = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ANKLE)

        def ratio(dist):
            return (dist / body_height) if dist else 0.25

        ratios = {
            'left_upper_arm': ratio(self._calculate_distance(l_shoulder, l_elbow)),
            'right_upper_arm': ratio(self._calculate_distance(r_shoulder, r_elbow)),
            'left_forearm': ratio(self._calculate_distance(l_elbow, l_wrist)),
            'right_forearm': ratio(self._calculate_distance(r_elbow, r_wrist)),
            'left_thigh': ratio(self._calculate_distance(l_hip, l_knee)),
            'right_thigh': ratio(self._calculate_distance(r_hip, r_knee)),
            'left_shin': ratio(self._calculate_distance(l_knee, l_ankle)),
            'right_shin': ratio(self._calculate_distance(r_knee, r_ankle)),
            'shoulder_width': ratio(self._calculate_distance(l_shoulder, r_shoulder)),
            'hip_width': ratio(self._calculate_distance(l_hip, r_hip)),
        }

        # Track confidence for each limb measurement
        confidences = {
            'left_upper_arm': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_LEFT_ELBOW]),
            'right_upper_arm': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_RIGHT_ELBOW]),
            'left_forearm': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_ELBOW, self.KEYPOINT_LEFT_WRIST]),
            'right_forearm': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_ELBOW, self.KEYPOINT_RIGHT_WRIST]),
            'left_thigh': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_HIP, self.KEYPOINT_LEFT_KNEE]),
            'right_thigh': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_HIP, self.KEYPOINT_RIGHT_KNEE]),
            'left_shin': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_KNEE, self.KEYPOINT_LEFT_ANKLE]),
            'right_shin': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_KNEE, self.KEYPOINT_RIGHT_ANKLE]),
            'shoulder_width': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_RIGHT_SHOULDER]),
            'hip_width': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_HIP, self.KEYPOINT_RIGHT_HIP]),
        }

        # Without a measurable body height (nose + an ankle), every ratio is
        # scaled by the 100px fallback — the values are scale-corrupted even
        # when their endpoint keypoints are confident. Zero the confidences so
        # masked search ignores them.
        if not height_valid:
            confidences = {k: 0.0 for k in confidences}

        return ratios, confidences

    def _extract_body_angles(self, keypoints: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract 15 body part orientation angles with confidence tracking.

        Returns:
            Tuple of (angles_dict, confidence_dict)
            - angles_dict: Body angle features (default 0° if missing)
            - confidence_dict: Confidence per angle (0.0 if any keypoint invalid)
        """
        # Get keypoints
        nose = self._get_keypoint(keypoints, self.KEYPOINT_NOSE)
        l_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_SHOULDER)
        r_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_SHOULDER)
        l_elbow = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ELBOW)
        r_elbow = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ELBOW)
        l_wrist = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_WRIST)
        r_wrist = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_WRIST)
        l_hip = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_HIP)
        r_hip = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_HIP)
        l_knee = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_KNEE)
        r_knee = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_KNEE)
        l_ankle = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ANKLE)
        r_ankle = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ANKLE)

        def angle_from_vertical(p1, p2):
            """Calculate angle from vertical axis."""
            if p1 is None or p2 is None:
                return 0.0
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return float(np.arctan2(dx, dy) * 180.0 / np.pi)

        # Calculate shoulder center and hip center
        shoulder_center = None
        if l_shoulder is not None and r_shoulder is not None:
            shoulder_center = (l_shoulder + r_shoulder) / 2

        hip_center = None
        if l_hip is not None and r_hip is not None:
            hip_center = (l_hip + r_hip) / 2

        angles = {
            'torso_lean': angle_from_vertical(hip_center, shoulder_center),
            'head_tilt': angle_from_vertical(shoulder_center, nose),
            'left_upper_arm_angle': angle_from_vertical(l_shoulder, l_elbow),
            'right_upper_arm_angle': angle_from_vertical(r_shoulder, r_elbow),
            'left_forearm_angle': angle_from_vertical(l_elbow, l_wrist),
            'right_forearm_angle': angle_from_vertical(r_elbow, r_wrist),
            'left_thigh_angle': angle_from_vertical(l_hip, l_knee),
            'right_thigh_angle': angle_from_vertical(r_hip, r_knee),
            'left_shin_angle': angle_from_vertical(l_knee, l_ankle),
            'right_shin_angle': angle_from_vertical(r_knee, r_ankle),
            'shoulder_line_angle': angle_from_vertical(l_shoulder, r_shoulder),
            'hip_line_angle': angle_from_vertical(l_hip, r_hip),
            'left_arm_spread': angle_from_vertical(l_shoulder, l_wrist),
            'right_arm_spread': angle_from_vertical(r_shoulder, r_wrist),
            'body_twist': 0.0  # Placeholder for torso rotation
        }

        # Track confidence for each orientation angle
        confidences = {
            'torso_lean': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_LEFT_HIP, self.KEYPOINT_RIGHT_HIP]),
            'head_tilt': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_NOSE, self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_RIGHT_SHOULDER]),
            'left_upper_arm_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_LEFT_ELBOW]),
            'right_upper_arm_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_RIGHT_ELBOW]),
            'left_forearm_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_ELBOW, self.KEYPOINT_LEFT_WRIST]),
            'right_forearm_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_ELBOW, self.KEYPOINT_RIGHT_WRIST]),
            'left_thigh_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_HIP, self.KEYPOINT_LEFT_KNEE]),
            'right_thigh_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_HIP, self.KEYPOINT_RIGHT_KNEE]),
            'left_shin_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_KNEE, self.KEYPOINT_LEFT_ANKLE]),
            'right_shin_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_KNEE, self.KEYPOINT_RIGHT_ANKLE]),
            'shoulder_line_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_RIGHT_SHOULDER]),
            'hip_line_angle': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_HIP, self.KEYPOINT_RIGHT_HIP]),
            'left_arm_spread': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_LEFT_WRIST]),
            'right_arm_spread': self._get_keypoint_confidence(keypoints, [self.KEYPOINT_RIGHT_SHOULDER, self.KEYPOINT_RIGHT_WRIST]),
            'body_twist': 0.0  # Placeholder, always 0 confidence
        }

        return angles, confidences

    def _extract_symmetry_scores(
        self,
        joint_angles: Dict[str, float],
        limb_ratios: Dict[str, float],
        angle_confidences: Dict[str, float],
        ratio_confidences: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract 8 left-right symmetry measures with confidence tracking.

        Args:
            joint_angles: Joint angle values
            limb_ratios: Limb ratio values
            angle_confidences: Confidence scores for joint angles
            ratio_confidences: Confidence scores for limb ratios

        Returns:
            Tuple of (symmetry_dict, confidence_dict)
            - symmetry_dict: Symmetry features (0 = symmetric, 1 = asymmetric)
            - confidence_dict: Confidence per symmetry (min of left & right confidences)
        """
        def symmetry(left_val, right_val, left_conf, right_conf):
            """Calculate symmetry score (0 = perfect symmetry).

            When either side is invalid (confidence 0), its value is an imputed
            default — comparing it against the real side produces a full-range
            garbage score. Return neutral 0.0 instead; the paired confidence
            (min of both sides) is already 0 so masked search ignores the dim.
            """
            if left_conf <= 0.0 or right_conf <= 0.0:
                return 0.0
            if left_val == 0 and right_val == 0:
                return 0.0
            return abs(left_val - right_val) / (abs(left_val) + abs(right_val) + 1e-6)

        scores = {
            'elbow_symmetry': symmetry(
                joint_angles['left_elbow'],
                joint_angles['right_elbow'],
                angle_confidences['left_elbow'],
                angle_confidences['right_elbow']
            ),
            'shoulder_symmetry': symmetry(
                joint_angles['left_shoulder'],
                joint_angles['right_shoulder'],
                angle_confidences['left_shoulder'],
                angle_confidences['right_shoulder']
            ),
            'hip_symmetry': symmetry(
                joint_angles['left_hip'],
                joint_angles['right_hip'],
                angle_confidences['left_hip'],
                angle_confidences['right_hip']
            ),
            'knee_symmetry': symmetry(
                joint_angles['left_knee'],
                joint_angles['right_knee'],
                angle_confidences['left_knee'],
                angle_confidences['right_knee']
            ),
            'upper_arm_length_symmetry': symmetry(
                limb_ratios['left_upper_arm'],
                limb_ratios['right_upper_arm'],
                ratio_confidences['left_upper_arm'],
                ratio_confidences['right_upper_arm']
            ),
            'forearm_length_symmetry': symmetry(
                limb_ratios['left_forearm'],
                limb_ratios['right_forearm'],
                ratio_confidences['left_forearm'],
                ratio_confidences['right_forearm']
            ),
            'thigh_length_symmetry': symmetry(
                limb_ratios['left_thigh'],
                limb_ratios['right_thigh'],
                ratio_confidences['left_thigh'],
                ratio_confidences['right_thigh']
            ),
            'shin_length_symmetry': symmetry(
                limb_ratios['left_shin'],
                limb_ratios['right_shin'],
                ratio_confidences['left_shin'],
                ratio_confidences['right_shin']
            ),
        }

        # Symmetry confidence = minimum of both sides (need both to compare)
        confidences = {
            'elbow_symmetry': min(
                angle_confidences['left_elbow'],
                angle_confidences['right_elbow']
            ),
            'shoulder_symmetry': min(
                angle_confidences['left_shoulder'],
                angle_confidences['right_shoulder']
            ),
            'hip_symmetry': min(
                angle_confidences['left_hip'],
                angle_confidences['right_hip']
            ),
            'knee_symmetry': min(
                angle_confidences['left_knee'],
                angle_confidences['right_knee']
            ),
            'upper_arm_length_symmetry': min(
                ratio_confidences['left_upper_arm'],
                ratio_confidences['right_upper_arm']
            ),
            'forearm_length_symmetry': min(
                ratio_confidences['left_forearm'],
                ratio_confidences['right_forearm']
            ),
            'thigh_length_symmetry': min(
                ratio_confidences['left_thigh'],
                ratio_confidences['right_thigh']
            ),
            'shin_length_symmetry': min(
                ratio_confidences['left_shin'],
                ratio_confidences['right_shin']
            ),
        }

        return scores, confidences

    def _extract_occlusion_pattern(
        self,
        keypoints: np.ndarray,
        visibility: np.ndarray
    ) -> np.ndarray:
        """
        Extract 7-dimensional occlusion pattern encoding.

        Each dimension is 1 if region is visible, 0 if occluded.
        Regions: head, left_arm, right_arm, torso, left_leg, right_leg, feet

        Args:
            keypoints: Keypoint array (133, 3)
            visibility: Visibility flags (133,)

        Returns:
            Binary array (7,) indicating region visibility
        """
        # Check visibility of major regions. >= 1.5 rather than == 2: same
        # semantics for integer visibility, robust to continuous values from
        # ensemble fusion or stored float arrays.
        def vis(idx):
            return float(visibility[idx]) >= 1.5

        head_visible = (
            vis(self.KEYPOINT_NOSE) or
            vis(self.KEYPOINT_LEFT_EYE) or
            vis(self.KEYPOINT_RIGHT_EYE)
        )

        left_arm_visible = (
            vis(self.KEYPOINT_LEFT_SHOULDER) and
            vis(self.KEYPOINT_LEFT_ELBOW)
        )

        right_arm_visible = (
            vis(self.KEYPOINT_RIGHT_SHOULDER) and
            vis(self.KEYPOINT_RIGHT_ELBOW)
        )

        torso_visible = (
            vis(self.KEYPOINT_LEFT_SHOULDER) and
            vis(self.KEYPOINT_RIGHT_SHOULDER) and
            vis(self.KEYPOINT_LEFT_HIP) and
            vis(self.KEYPOINT_RIGHT_HIP)
        )

        left_leg_visible = (
            vis(self.KEYPOINT_LEFT_HIP) and
            vis(self.KEYPOINT_LEFT_KNEE)
        )

        right_leg_visible = (
            vis(self.KEYPOINT_RIGHT_HIP) and
            vis(self.KEYPOINT_RIGHT_KNEE)
        )

        feet_visible = (
            vis(self.KEYPOINT_LEFT_ANKLE) or
            vis(self.KEYPOINT_RIGHT_ANKLE)
        )

        return np.array([
            float(head_visible),
            float(left_arm_visible),
            float(right_arm_visible),
            float(torso_visible),
            float(left_leg_visible),
            float(right_leg_visible),
            float(feet_visible)
        ], dtype=np.float32)

    def _build_feature_vector(
        self,
        joint_angles: Dict[str, float],
        limb_ratios: Dict[str, float],
        body_angles: Dict[str, float],
        symmetry_scores: Dict[str, float],
        occlusion_pattern: np.ndarray,
        angle_confidences: Dict[str, float],
        ratio_confidences: Dict[str, float],
        body_angle_confidences: Dict[str, float],
        symmetry_confidences: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine all features and confidences into 52-dimensional vectors.

        Returns:
            Tuple of (feature_vector, feature_confidence)
            - feature_vector: (52,) feature values as float32
            - feature_confidence: (52,) confidence scores 0-1 as float32
        """
        # Concatenate all features in order
        features = []
        confidences = []

        # Joint angles (12)
        features.extend([
            joint_angles['left_elbow'], joint_angles['right_elbow'],
            joint_angles['left_shoulder'], joint_angles['right_shoulder'],
            joint_angles['left_hip'], joint_angles['right_hip'],
            joint_angles['left_knee'], joint_angles['right_knee'],
            joint_angles['left_armpit'], joint_angles['right_armpit'],
            joint_angles['left_leg_spread'], joint_angles['right_leg_spread']
        ])
        confidences.extend([
            angle_confidences['left_elbow'], angle_confidences['right_elbow'],
            angle_confidences['left_shoulder'], angle_confidences['right_shoulder'],
            angle_confidences['left_hip'], angle_confidences['right_hip'],
            angle_confidences['left_knee'], angle_confidences['right_knee'],
            angle_confidences['left_armpit'], angle_confidences['right_armpit'],
            angle_confidences['left_leg_spread'], angle_confidences['right_leg_spread']
        ])

        # Limb ratios (10)
        features.extend([
            limb_ratios['left_upper_arm'], limb_ratios['right_upper_arm'],
            limb_ratios['left_forearm'], limb_ratios['right_forearm'],
            limb_ratios['left_thigh'], limb_ratios['right_thigh'],
            limb_ratios['left_shin'], limb_ratios['right_shin'],
            limb_ratios['shoulder_width'], limb_ratios['hip_width']
        ])
        confidences.extend([
            ratio_confidences['left_upper_arm'], ratio_confidences['right_upper_arm'],
            ratio_confidences['left_forearm'], ratio_confidences['right_forearm'],
            ratio_confidences['left_thigh'], ratio_confidences['right_thigh'],
            ratio_confidences['left_shin'], ratio_confidences['right_shin'],
            ratio_confidences['shoulder_width'], ratio_confidences['hip_width']
        ])

        # Body angles (15)
        features.extend([
            body_angles['torso_lean'], body_angles['head_tilt'],
            body_angles['left_upper_arm_angle'], body_angles['right_upper_arm_angle'],
            body_angles['left_forearm_angle'], body_angles['right_forearm_angle'],
            body_angles['left_thigh_angle'], body_angles['right_thigh_angle'],
            body_angles['left_shin_angle'], body_angles['right_shin_angle'],
            body_angles['shoulder_line_angle'], body_angles['hip_line_angle'],
            body_angles['left_arm_spread'], body_angles['right_arm_spread'],
            body_angles['body_twist']
        ])
        confidences.extend([
            body_angle_confidences['torso_lean'], body_angle_confidences['head_tilt'],
            body_angle_confidences['left_upper_arm_angle'], body_angle_confidences['right_upper_arm_angle'],
            body_angle_confidences['left_forearm_angle'], body_angle_confidences['right_forearm_angle'],
            body_angle_confidences['left_thigh_angle'], body_angle_confidences['right_thigh_angle'],
            body_angle_confidences['left_shin_angle'], body_angle_confidences['right_shin_angle'],
            body_angle_confidences['shoulder_line_angle'], body_angle_confidences['hip_line_angle'],
            body_angle_confidences['left_arm_spread'], body_angle_confidences['right_arm_spread'],
            body_angle_confidences['body_twist']
        ])

        # Symmetry scores (8)
        features.extend([
            symmetry_scores['elbow_symmetry'], symmetry_scores['shoulder_symmetry'],
            symmetry_scores['hip_symmetry'], symmetry_scores['knee_symmetry'],
            symmetry_scores['upper_arm_length_symmetry'], symmetry_scores['forearm_length_symmetry'],
            symmetry_scores['thigh_length_symmetry'], symmetry_scores['shin_length_symmetry']
        ])
        confidences.extend([
            symmetry_confidences['elbow_symmetry'], symmetry_confidences['shoulder_symmetry'],
            symmetry_confidences['hip_symmetry'], symmetry_confidences['knee_symmetry'],
            symmetry_confidences['upper_arm_length_symmetry'], symmetry_confidences['forearm_length_symmetry'],
            symmetry_confidences['thigh_length_symmetry'], symmetry_confidences['shin_length_symmetry']
        ])

        # Occlusion pattern (7) - these are visibility flags, always confidence = 1.0
        features.extend(occlusion_pattern.tolist())
        confidences.extend([1.0] * 7)  # Occlusion pattern always confident (represents presence)

        vector = np.array(features, dtype=np.float32)
        confidence_vector = np.array(confidences, dtype=np.float32)

        # Normalize to [0, 1] if requested. Each block gets its own transform —
        # the old single `vector[:45] /= 180` also divided the limb-ratio and
        # symmetry dims (already ~[0,1]) by 180, crushing 18 of 52 dims to
        # numerical irrelevance, and clipped every negative body angle to 0.
        if self.normalize:
            # Joint angles (0-11): 0..180 degrees
            vector[0:12] = np.clip(vector[0:12] / 180.0, 0.0, 1.0)
            # Limb ratios (12-21): body-height fractions, already ~[0,1]
            vector[12:22] = np.clip(vector[12:22], 0.0, 1.0)
            # Body angles (22-36): signed -180..180 from vertical; map so the
            # sign survives (0 degrees -> 0.5, the missing-keypoint default)
            vector[22:37] = np.clip((vector[22:37] + 180.0) / 360.0, 0.0, 1.0)
            # Symmetry (37-44): already [0,1] by construction
            # Occlusion flags (45-51): binary; down-weighted so one mismatched
            # flag costs the same squared distance as a 45-degree joint-angle
            # difference instead of a 180-degree one
            vector[45:52] = vector[45:52] * self.OCCLUSION_FLAG_WEIGHT

        return vector, confidence_vector

    def extract(
        self,
        pose_result: PoseResult,
        viewpoint: Optional[dict] = None
    ) -> GeometricFeatures:
        """
        Extract complete geometric features from pose.

        Args:
            pose_result: PoseResult from pose detector
            viewpoint: Optional viewpoint estimate (not currently used but reserved)

        Returns:
            GeometricFeatures with 52-dim vector and component features
        """
        keypoints = pose_result.keypoints
        visibility = pose_result.visibility

        # Store visibility for _get_keypoint() to access
        self._current_visibility = visibility

        logger.debug(
            f"Extracting features from pose {pose_result.person_id}",
            extra={'extra_data': {
                'visible_keypoints': pose_result.count_visible(),
                'occluded_keypoints': pose_result.count_occluded(),
                'use_occluded': self.use_occluded_keypoints
            }}
        )

        # Extract all feature components with confidence tracking
        joint_angles, angle_confidences = self._extract_joint_angles(keypoints)
        limb_ratios, ratio_confidences = self._extract_limb_ratios(keypoints)
        body_angles, body_angle_confidences = self._extract_body_angles(keypoints)
        symmetry_scores, symmetry_confidences = self._extract_symmetry_scores(
            joint_angles, limb_ratios, angle_confidences, ratio_confidences
        )
        occlusion_pattern = self._extract_occlusion_pattern(keypoints, visibility)

        # Build feature vector and confidence vector
        feature_vector, feature_confidence = self._build_feature_vector(
            joint_angles,
            limb_ratios,
            body_angles,
            symmetry_scores,
            occlusion_pattern,
            angle_confidences,
            ratio_confidences,
            body_angle_confidences,
            symmetry_confidences
        )

        features = GeometricFeatures(
            feature_vector=feature_vector,
            feature_confidence=feature_confidence,
            joint_angles=joint_angles,
            limb_ratios=limb_ratios,
            body_angles=body_angles,
            symmetry_scores=symmetry_scores,
            occlusion_pattern=occlusion_pattern
        )

        logger.info(
            f"Features extracted for pose {pose_result.person_id}",
            extra={'extra_data': {
                'feature_dim': len(feature_vector),
                'joint_angles_count': len(joint_angles),
                'limb_ratios_count': len(limb_ratios)
            }}
        )

        return features

    def flip_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Flip feature vector horizontally by swapping left/right features.

        This enables mirror-invariant pose matching (e.g., left-facing finds right-facing).

        Args:
            feature_vector: 52-dimensional feature vector to flip

        Returns:
            Flipped 52-dimensional feature vector with left/right swapped

        Feature breakdown (52 dimensions total):
        - Joint angles (12): indices 0-11
        - Limb ratios (10): indices 12-21
        - Body angles (15): indices 22-36
        - Symmetry scores (8): indices 37-44
        - Occlusion pattern (7): indices 45-51
        """
        if len(feature_vector) != 52:
            raise ValueError(f"Expected 52-dimensional vector, got {len(feature_vector)}")

        # Create copy to avoid modifying original
        flipped = feature_vector.copy()

        # Swap left/right joint angles (indices 0-11)
        # Order: left_elbow, right_elbow, left_shoulder, right_shoulder,
        #        left_hip, right_hip, left_knee, right_knee,
        #        left_armpit, right_armpit, left_leg_spread, right_leg_spread
        angle_swaps = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        for left_idx, right_idx in angle_swaps:
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]

        # Swap left/right limb ratios (indices 12-21)
        # Order: left_upper_arm, right_upper_arm, left_forearm, right_forearm,
        #        left_thigh, right_thigh, left_shin, right_shin,
        #        shoulder_width, hip_width (last 2 are symmetric, no swap)
        ratio_swaps = [(12, 13), (14, 15), (16, 17), (18, 19)]  # 4 pairs only
        for left_idx, right_idx in ratio_swaps:
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]
        # Indices 20-21 (shoulder_width, hip_width) are symmetric - no swap

        # Swap left/right body angles (indices 22-36)
        # Order: torso_lean, head_tilt (symmetric),
        #        left_upper_arm_angle, right_upper_arm_angle, left_forearm_angle, right_forearm_angle,
        #        left_thigh_angle, right_thigh_angle, left_shin_angle, right_shin_angle,
        #        shoulder_line_angle, hip_line_angle (symmetric),
        #        left_arm_spread, right_arm_spread, body_twist (symmetric)
        body_angle_swaps = [(24, 25), (26, 27), (28, 29), (30, 31), (34, 35)]  # 5 pairs
        for left_idx, right_idx in body_angle_swaps:
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]
        # Indices 22-23, 32-33, 36 are symmetric (torso, lines, twist) - no swap

        # Symmetry scores (indices 37-44) - NO SWAP
        # These already measure left-right balance, swapping would be incorrect
        # They stay the same because symmetry is symmetric!

        # Swap left/right occlusion pattern (indices 45-51)
        # Order: head (symmetric), left_arm, right_arm, torso (symmetric),
        #        left_leg, right_leg, feet (symmetric)
        occlusion_swaps = [(46, 47), (49, 50)]  # 2 pairs only
        for left_idx, right_idx in occlusion_swaps:
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]
        # Indices 45 (head), 48 (torso), 51 (feet) are symmetric - no swap

        return flipped
