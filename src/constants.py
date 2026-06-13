"""
Constants for PostureKit to replace magic numbers and improve code readability.
"""

# Keypoint dimensions
RTMW_KEYPOINTS = 133
COCO_KEYPOINTS = 17
KEYPOINT_CHANNELS = 3  # x, y, confidence
KEYPOINTS_SIZE = RTMW_KEYPOINTS * KEYPOINT_CHANNELS  # 399
COCO_KEYPOINTS_SIZE = COCO_KEYPOINTS * KEYPOINT_CHANNELS  # 51

# Bounding box
BBOX_DIMENSIONS = 4  # x, y, width, height
MIN_BBOX_SIZE_PX = 50  # Minimum bbox size in pixels

# Confidence thresholds
MIN_KEYPOINT_CONFIDENCE = 0.3
MIN_PERSON_CONFIDENCE = 0.5
MIN_FEATURE_CONFIDENCE = 0.35

# Bbox refinement
BBOX_PADDING_BASE = 0.05
BBOX_PADDING_OCCLUSION_FACTOR = 0.10

# Feature dimensions (geometric vector format v3)
GEOMETRIC_FEATURE_DIM = 66
VISUAL_FEATURE_DIM = 576
FUSED_FEATURE_DIM = GEOMETRIC_FEATURE_DIM + VISUAL_FEATURE_DIM  # 642

# Geometric feature composition (v3)
NUM_JOINT_ANGLES = 12
NUM_LIMB_RATIOS = 10
NUM_BODY_ANGLE_SINCOS = 28      # 14 signed angles x [sin_part, cos_part]
NUM_BODY_TWIST = 1             # non-wrapping shoulder-vs-hip torsion cue
NUM_SYMMETRY_FEATURES = 8
NUM_OCCLUSION_FEATURES = 7

# OKS (Object Keypoint Similarity)
MIN_DISPLACEMENT_PX = 1.0  # Minimum pixel displacement to count as correction

# Body part detection
HAND_KEYPOINTS_PER_HAND = 21
TOTAL_HAND_KEYPOINTS = HAND_KEYPOINTS_PER_HAND * 2  # 42

# Left/right keypoint pairs for horizontal flipping
LEFT_EYE_IDX = 1
RIGHT_EYE_IDX = 2
LEFT_EAR_IDX = 3
RIGHT_EAR_IDX = 4

# Search and indexing
DEFAULT_SEARCH_K = 20
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL_SECONDS = 600  # 10 minutes
BATCH_PROCESSING_SIZE = 1000

# Distance and similarity
# Default exp(-d/scale) base when the index mapping metadata carries no fitted
# 'similarity_scale' (e.g. legacy indices). The live value is read from index
# metadata so Wave 4 can fit it post-re-extraction without code changes; see
# SimilarityEngine._distance_to_similarity / save_index.
DEFAULT_SIMILARITY_SCALE = 2.0
REFERENCE_FEATURE_DIMENSION = 52.0

# Plausibility scoring
MIN_LIMB_SYMMETRY = 0.5
SYMMETRY_PENALTY_FACTOR = 0.5
PROPORTION_RATIO_MIN = 0.7
PROPORTION_RATIO_MAX = 1.3

# Threading
DEFAULT_NUM_WORKERS = 3
DEFAULT_DEBOUNCE_SECONDS = 2.0

# Image processing
DEFAULT_THUMBNAIL_SIZE = 200
DEFAULT_THUMBNAIL_QUALITY = 80

# Visibility flags (COCO format)
VISIBILITY_NOT_LABELED = 0
VISIBILITY_LABELED_OCCLUDED = 1
VISIBILITY_LABELED_VISIBLE = 2

# Visibility classification thresholds
VISIBILITY_CONFIDENCE_VISIBLE = 0.5      # Keypoint visible (visibility=2)
VISIBILITY_CONFIDENCE_OCCLUDED = 0.1    # Keypoint occluded (visibility=1)

# Body part detection constraints
MIN_BODY_PART_BBOX_SIZE = 30            # Minimum bbox dimension for body part detection
BODY_PART_DETECTION_MARGIN = 50         # Margin pixels for part region search

# Plausibility scoring weights
PLAUSIBILITY_WEIGHT_SYMMETRY = 0.40     # 40% weight for symmetry score
PLAUSIBILITY_WEIGHT_ANGLES = 0.30       # 30% weight for angle validity
PLAUSIBILITY_WEIGHT_PROPORTIONS = 0.30  # 30% weight for proportion consistency
INVALID_ANGLE_SCORE = 0.5               # Fallback score for anatomically invalid angles
UNUSUAL_PROPORTION_PENALTY = 0.7        # Penalty for unusual body proportions

# Duplicate detection
MIN_OVERLAP_RATIO = 0.5                 # 50% overlap = duplicate

# Progress reporting
PROGRESS_REPORT_INTERVAL = 50           # Report progress every N images
