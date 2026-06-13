"""
Feature Extractor for PostureKit.
Computes 66-dimensional geometric feature vectors from pose keypoints.

Vector format v3 (66 dims): the 14 signed body-orientation angles are encoded
as continuous sin/cos pairs (28 dims) to remove the +-180 wraparound that put
upright torso/head exactly on a discontinuity; the former dead body_twist
placeholder is replaced by a real, non-wrapping shoulder-vs-hip torsion cue.
Re-extraction (scripts/reextract_features.py) is REQUIRED per DB_PROFILE after
adopting this layout — stored v2 (52-dim) vectors are incomparable with v3.
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
        feature_vector: 66-dimensional feature array
        feature_confidence: 66-dimensional confidence array (0-1 per feature)
        joint_angles: Dictionary of computed joint angles
        limb_ratios: Dictionary of limb length ratios
        body_angles: Dictionary of body part orientations
        symmetry_scores: Dictionary of left-right symmetry measures
        occlusion_pattern: Binary array indicating visible keypoints
    """
    feature_vector: np.ndarray  # (66,)
    feature_confidence: np.ndarray  # (66,) confidence scores 0-1
    joint_angles: Dict[str, float]
    limb_ratios: Dict[str, float]
    body_angles: Dict[str, float]
    symmetry_scores: Dict[str, float]
    occlusion_pattern: np.ndarray  # (7,)

    def __post_init__(self):
        """Validate feature dimensions."""
        assert self.feature_vector.shape == (66,), \
            f"Feature vector must be (66,), got {self.feature_vector.shape}"
        assert self.feature_vector.dtype == np.float32, \
            f"Feature vector must be float32, got {self.feature_vector.dtype}"
        assert self.feature_confidence.shape == (66,), \
            f"Feature confidence must be (66,), got {self.feature_confidence.shape}"
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
    Extracts 66-dimensional geometric feature vectors from poses (format v3).

    Feature breakdown (66 dimensions total):
    - Joint angles (12, dims 0-11): elbow, knee, shoulder, hip angles
    - Limb length ratios (10, dims 12-21): normalized by body height
    - Body-angle sin/cos pairs (28, dims 22-49): 14 signed orientation angles,
      each encoded as [sin_part, cos_part] so the metric is continuous across
      the +-180 boundary where upright torso/head used to sit.
    - body_twist (1, dim 50): non-wrapping shoulder-vs-hip torsion cue.
    - Symmetry scores (8, dims 51-58): left vs right body parts
    - Occlusion pattern (7, dims 59-65): encoded visibility of major body regions

    Attributes:
        confidence_threshold: Minimum confidence for valid keypoint
        normalize: Whether to normalize features to [0, 1] range
    """

    # Order of the 14 signed body-orientation angles that get sin/cos-encoded
    # into dims 22-49 (two dims per angle, [sin_part, cos_part]). Must match the
    # body-angle dict keys in _extract_body_angles. body_twist is NOT in this
    # list — it is a separate non-angular cue at dim 50.
    SINCOS_BODY_ANGLE_KEYS = [
        'torso_lean', 'head_tilt',
        'left_upper_arm_angle', 'right_upper_arm_angle',
        'left_forearm_angle', 'right_forearm_angle',
        'left_thigh_angle', 'right_thigh_angle',
        'left_shin_angle', 'right_shin_angle',
        'shoulder_line_angle', 'hip_line_angle',
        'left_arm_spread', 'right_arm_spread',
    ]

    # sin/cos parts are scaled by 1/sqrt(2) so the two-dim pair for one angle
    # spans the same squared-L2 budget (max 1.0) a single normalized angle dim
    # did under the old (a+180)/360 encoding. (sin+1)/2 and (cos+1)/2 land in
    # [0,1]; * SINCOS_SCALE puts them in [0, 1/sqrt(2)] ~ [0, 0.707].
    SINCOS_SCALE = 1.0 / np.sqrt(2.0)

    # Max absolute shoulder-vs-hip torsion (degrees) before the body_twist cue
    # saturates. Beyond ~90 deg the two lines are effectively perpendicular;
    # clipping here keeps the cue bounded and non-wrapping (see _build path).
    BODY_TWIST_MAX_DEG = 90.0

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

    # Anatomical constant: full nose<->ankle body height is ~2.4x the
    # shoulder-center<->hip-center torso length (used by the fallback anchor).
    TORSO_TO_HEIGHT_RATIO = 2.4

    def _calculate_body_height(
        self,
        keypoints: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Calculate a body-height scale anchor with a graceful fallback chain.

        Limb ratios are normalized by this scale, so a missing primary anchor
        must NOT zero the ratios (finding 17): head-cropped portraits and
        legs-out-of-frame shots are common, and their endpoint keypoints are
        often confident. Instead we fall back to progressively weaker anchors
        and report an anchor-quality factor the caller uses to discount (not
        zero) the limb-ratio confidences.

        Fallback chain:
          1. nose<->ankle vertical extent (best)            -> quality 1.0
          2. torso length (shoulder_center<->hip_center)
             * TORSO_TO_HEIGHT_RATIO (reliably visible)      -> quality 0.8
          3. bbox diagonal (last resort)                     -> quality 0.5

        Returns:
            (body_height_px, anchor_quality). When no anchor at all is
            available, returns (100.0, 0.0) so the caller zeros the ratios.
        """
        # Anchor 1: nose <-> ankle
        nose = self._get_keypoint(keypoints, self.KEYPOINT_NOSE)
        l_ankle = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_ANKLE)
        r_ankle = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_ANKLE)

        heights = []
        if nose is not None and l_ankle is not None:
            heights.append(self._calculate_distance(nose, l_ankle))
        if nose is not None and r_ankle is not None:
            heights.append(self._calculate_distance(nose, r_ankle))
        if heights:
            return float(np.mean(heights)), 1.0

        # Anchor 2: torso length (shoulder center <-> hip center) * constant
        l_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_SHOULDER)
        r_shoulder = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_SHOULDER)
        l_hip = self._get_keypoint(keypoints, self.KEYPOINT_LEFT_HIP)
        r_hip = self._get_keypoint(keypoints, self.KEYPOINT_RIGHT_HIP)
        if (l_shoulder is not None and r_shoulder is not None and
                l_hip is not None and r_hip is not None):
            shoulder_center = (l_shoulder + r_shoulder) / 2
            hip_center = (l_hip + r_hip) / 2
            torso_len = self._calculate_distance(shoulder_center, hip_center)
            if torso_len and torso_len > 1e-6:
                return float(torso_len * self.TORSO_TO_HEIGHT_RATIO), 0.8

        # Anchor 3: bbox diagonal. PoseResult.bbox is [x, y, w, h] (xywh), so
        # width/height are bx[2]/bx[3] directly.
        if bbox is not None:
            bx = np.asarray(bbox, dtype=np.float32).ravel()
            if bx.shape[0] >= 4:
                w = float(bx[2])
                h = float(bx[3])
                diag = float(np.hypot(w, h))
                if diag > 1e-6:
                    return diag, 0.5

        # No usable anchor at all
        return 100.0, 0.0

    def _extract_limb_ratios(
        self,
        keypoints: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract 10 limb length ratios normalized by body height with confidence tracking.

        Returns:
            Tuple of (ratios_dict, confidence_dict)
            - ratios_dict: Limb ratio features (default 0.25 if missing)
            - confidence_dict: Confidence per ratio (0.0 if any endpoint invalid)
        """
        body_height, anchor_quality = self._calculate_body_height(keypoints, bbox)

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

        # Scale ratio confidences by the anchor quality rather than zeroing
        # them when the primary nose<->ankle anchor is unavailable (finding 17).
        # A torso-length (0.8) or bbox-diagonal (0.5) anchor still yields a
        # usable scale for the very common legs-or-face-out-of-frame crops, so
        # the 10 ratio dims stay populated with appropriately reduced trust.
        # anchor_quality == 0.0 (no anchor at all) still zeros them.
        if anchor_quality < 1.0:
            confidences = {k: v * anchor_quality for k, v in confidences.items()}

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
        }

        # body_twist (finding 19): a REAL, non-wrapping torso-rotation cue from
        # the most reliably visible keypoints (shoulders + hips). It is the
        # absolute angular difference between the shoulder line and the hip
        # line — a counter-tilt / contrapposto signal — folded into [0,180] so
        # it never wraps (a line and its 180-deg reverse are the same line),
        # then clipped to BODY_TWIST_MAX_DEG and normalized to [0,1]. This is a
        # BOUNDED RATIO, not a raw angle, so it does not reintroduce the
        # +-180 discontinuity that motivated the sin/cos encoding. 0.0 means
        # shoulder and hip lines are parallel (no torsion); 1.0 means they are
        # >= BODY_TWIST_MAX_DEG apart. Requires all 4 keypoints; otherwise 0.0
        # value with 0.0 confidence (see confidences dict below).
        body_twist = 0.0
        if (l_shoulder is not None and r_shoulder is not None and
                l_hip is not None and r_hip is not None):
            shoulder_line = angles['shoulder_line_angle']
            hip_line = angles['hip_line_angle']
            raw_diff = abs(shoulder_line - hip_line)
            # Fold so a line and its reversed twin are equivalent: map the
            # difference into [0, 180], then take the acute representation
            # [0, 90] (torsion is direction-agnostic), giving a continuous,
            # wrap-free magnitude.
            folded = raw_diff % 360.0
            if folded > 180.0:
                folded = 360.0 - folded
            if folded > 90.0:
                folded = 180.0 - folded
            body_twist = float(min(folded, self.BODY_TWIST_MAX_DEG) / self.BODY_TWIST_MAX_DEG)
        angles['body_twist'] = body_twist

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
            # body_twist (finding 19): now a real cue. Confidence is the minimum
            # contribution of the 4 required keypoints (both shoulders + both
            # hips); _get_keypoint_confidence already returns 0.0 if any of them
            # fails the visibility/confidence gate.
            'body_twist': self._get_keypoint_confidence(keypoints, [
                self.KEYPOINT_LEFT_SHOULDER, self.KEYPOINT_RIGHT_SHOULDER,
                self.KEYPOINT_LEFT_HIP, self.KEYPOINT_RIGHT_HIP
            ]),
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
        Combine all features and confidences into 66-dimensional vectors (v3).

        Layout (66 dims):
          [ 0:12)  joint angles            (12)
          [12:22)  limb ratios             (10)
          [22:50)  body-angle sin/cos pairs (28) = 14 angles x [sin_part, cos_part]
          [50]     body_twist               (1)
          [51:59)  symmetry scores          (8)
          [59:66)  occlusion flags          (7)

        Returns:
            Tuple of (feature_vector, feature_confidence)
            - feature_vector: (66,) feature values as float32
            - feature_confidence: (66,) confidence scores 0-1 as float32
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

        # Body-angle sin/cos pairs (28 dims = 14 angles x [sin_part, cos_part]).
        # Encoding each signed angle a (degrees) as a continuous (sin, cos) pair
        # removes the +-180 wraparound where upright torso/head used to sit
        # (finding 16). For each angle:
        #   sin_part = (sin(rad(a)) + 1)/2 * SINCOS_SCALE   in [0, 1/sqrt(2)]
        #   cos_part = (cos(rad(a)) + 1)/2 * SINCOS_SCALE   in [0, 1/sqrt(2)]
        # SINCOS_SCALE = 1/sqrt(2) keeps the per-angle squared-L2 budget at 1.0,
        # matching the old single normalized angle dim. The angle's existing
        # confidence is DUPLICATED into both the sin and cos slot so confidence
        # stays aligned 1:1 with the vector (load-bearing for masked search).
        # Missing-angle default is 0 deg -> sin_part=0.354, cos_part=0.707,
        # confidence 0.
        for key in self.SINCOS_BODY_ANGLE_KEYS:
            a_rad = np.radians(body_angles[key])
            sin_part = (np.sin(a_rad) + 1.0) / 2.0 * self.SINCOS_SCALE
            cos_part = (np.cos(a_rad) + 1.0) / 2.0 * self.SINCOS_SCALE
            features.extend([float(sin_part), float(cos_part)])
            conf = body_angle_confidences[key]
            confidences.extend([conf, conf])

        # body_twist (1 dim): real non-wrapping shoulder-vs-hip torsion cue,
        # already a bounded [0,1] ratio at extraction time (finding 19).
        features.append(body_angles['body_twist'])
        confidences.append(body_angle_confidences['body_twist'])

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

        # Normalize to [0, 1] if requested. Each block gets its own transform.
        # v3 layout (66 dims):
        #   [ 0:12) joint angles, [12:22) limb ratios, [22:50) sin/cos pairs,
        #   [50] body_twist, [51:59) symmetry, [59:66) occlusion flags.
        if self.normalize:
            # Joint angles (0-11): 0..180 degrees -> /180
            vector[0:12] = np.clip(vector[0:12] / 180.0, 0.0, 1.0)
            # Limb ratios (12-21): body-height fractions, already ~[0,1]
            vector[12:22] = np.clip(vector[12:22], 0.0, 1.0)
            # Body-angle sin/cos pairs (22-49) AND body_twist (50) are ALREADY
            # normalized to [0,1] at assembly time (sin/cos parts live in
            # [0, 1/sqrt(2)]; body_twist in [0,1]). Do NOT re-transform them —
            # the old (x+180)/360 mapping no longer applies. They are already in
            # range by construction; the assert below documents that invariant.
            assert np.all(vector[22:51] >= -1e-6) and np.all(vector[22:51] <= 1.0 + 1e-6), \
                "sin/cos + body_twist dims (22:51) must already be in [0,1]"
            # Symmetry (51-58): already [0,1] by construction — untouched
            # Occlusion flags (59-65): binary; down-weighted so one mismatched
            # flag costs the same squared distance as a ~27-degree joint-angle
            # difference instead of a 180-degree one (see OCCLUSION_FLAG_WEIGHT)
            vector[59:66] = vector[59:66] * self.OCCLUSION_FLAG_WEIGHT

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
            GeometricFeatures with 66-dim vector and component features
        """
        keypoints = pose_result.keypoints
        visibility = pose_result.visibility
        bbox = getattr(pose_result, 'bbox', None)

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
        limb_ratios, ratio_confidences = self._extract_limb_ratios(keypoints, bbox)
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
        Flip feature vector horizontally by swapping left/right features (v3).

        This enables mirror-invariant pose matching (e.g., left-facing finds right-facing).

        v3 layout (66 dims):
        - Joint angles (12): indices 0-11
        - Limb ratios (10): indices 12-21
        - Body-angle sin/cos pairs (28): indices 22-49 (14 angles x [sin, cos])
        - body_twist (1): index 50
        - Symmetry scores (8): indices 51-58
        - Occlusion pattern (7): indices 59-65

        Body-angle handling under a horizontal mirror (v3-specific):
        Each signed angle a is measured as atan2(dx, dy) from vertical, so a
        horizontal flip negates it (dx -> -dx => a -> -a). Under negation:
            sin(-a) = -sin(a)  ->  sin_part' = SINCOS_SCALE - sin_part
            cos(-a) =  cos(a)  ->  cos_part unchanged
        So EVERY sin/cos angle pair has its sin_part reflected (the old 52-dim
        flip only swapped and never negated signed angles — a latent bug flagged
        in finding 16). Left/right limb angles additionally swap their pairs.

        Args:
            feature_vector: 66-dimensional feature vector to flip

        Returns:
            Flipped 66-dimensional feature vector with left/right swapped
        """
        if len(feature_vector) != 66:
            raise ValueError(f"Expected 66-dimensional vector, got {len(feature_vector)}")

        # Create copy to avoid modifying original
        flipped = feature_vector.copy()

        # --- Joint angles (indices 0-11): swap L/R ---
        # Order: left_elbow, right_elbow, left_shoulder, right_shoulder,
        #        left_hip, right_hip, left_knee, right_knee,
        #        left_armpit, right_armpit, left_leg_spread, right_leg_spread
        angle_swaps = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        for left_idx, right_idx in angle_swaps:
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]

        # --- Limb ratios (indices 12-21): swap L/R ---
        # Order: left_upper_arm, right_upper_arm, left_forearm, right_forearm,
        #        left_thigh, right_thigh, left_shin, right_shin,
        #        shoulder_width, hip_width (last 2 symmetric, no swap)
        ratio_swaps = [(12, 13), (14, 15), (16, 17), (18, 19)]
        for left_idx, right_idx in ratio_swaps:
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]
        # Indices 20-21 (shoulder_width, hip_width) are symmetric - no swap

        # --- Body-angle sin/cos pairs (indices 22-49) ---
        # Step 1: reflect the sin_part of EVERY angle pair (a -> -a). The cos_part
        # is invariant under negation. Pair i occupies (sin=22+2i, cos=23+2i).
        for i in range(len(self.SINCOS_BODY_ANGLE_KEYS)):
            sin_idx = 22 + 2 * i
            flipped[sin_idx] = self.SINCOS_SCALE - flipped[sin_idx]
            # cos at 23+2i unchanged

        # Step 2: swap the (sin, cos) pairs of left/right limb angles. Index i in
        # SINCOS_BODY_ANGLE_KEYS:
        #   0 torso_lean, 1 head_tilt (self-symmetric, no swap),
        #   2 L_upper_arm <-> 3 R_upper_arm, 4 L_forearm <-> 5 R_forearm,
        #   6 L_thigh <-> 7 R_thigh, 8 L_shin <-> 9 R_shin,
        #   10 shoulder_line, 11 hip_line (self-symmetric, no swap),
        #   12 L_arm_spread <-> 13 R_arm_spread.
        sincos_pair_swaps = [(2, 3), (4, 5), (6, 7), (8, 9), (12, 13)]
        for li, ri in sincos_pair_swaps:
            l_base, r_base = 22 + 2 * li, 22 + 2 * ri
            # swap sin
            flipped[l_base], flipped[r_base] = flipped[r_base], flipped[l_base]
            # swap cos
            flipped[l_base + 1], flipped[r_base + 1] = flipped[r_base + 1], flipped[l_base + 1]

        # --- body_twist (index 50): mirror-invariant (folded |shoulder-hip|) - no change ---

        # --- Symmetry scores (indices 51-58): NO SWAP ---
        # These already measure left-right balance; swapping would be incorrect.

        # --- Occlusion pattern (indices 59-65): swap L/R ---
        # Order (offset +59): head (symmetric), left_arm, right_arm,
        #        torso (symmetric), left_leg, right_leg, feet (symmetric)
        occlusion_swaps = [(60, 61), (63, 64)]
        for left_idx, right_idx in occlusion_swaps:
            flipped[left_idx], flipped[right_idx] = flipped[right_idx], flipped[left_idx]
        # Indices 59 (head), 62 (torso), 65 (feet) are symmetric - no swap

        return flipped
